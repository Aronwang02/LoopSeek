"""Bounded CPU cache for the frozen Qwen3-VL image/video tower, including DeepStack.

Keys identify the exact processed media and grid in one reader instance. Language
positions, timestamps, prompts and generation still go through the normal model.
"""
from collections import OrderedDict
import hashlib

import torch


class VisionFeatureCache:
    def __init__(self, model, capacity_mb):
        self.original = model.get_image_features
        self.merge = model.visual.spatial_merge_size ** 2
        self.capacity = int(capacity_mb * 1024 ** 2)
        self.entries = OrderedDict()
        self.bytes = 0
        self.requests = {}
        self.hits = self.misses = 0
        model.get_image_features = self.get_features

    def register(self, cpu_inputs, gpu_inputs):
        self.requests = {}
        for pixels, grid in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
            if pixels not in cpu_inputs:
                continue
            values = cpu_inputs[pixels]
            grids = cpu_inputs[grid].tolist()
            keys, offset = [], 0
            for shape in grids:
                size = shape[0] * shape[1] * shape[2]
                # Hash the actual normalized pixels, not an image id that could
                # silently alias different resizing / video temporal patches.
                chunk = values[offset:offset + size].contiguous()
                digest = hashlib.blake2b(chunk.view(torch.uint8).numpy(), digest_size=20).digest()
                keys.append((tuple(shape), str(chunk.dtype), digest))
                offset += size
            self.requests[gpu_inputs[pixels].data_ptr()] = (keys, grids)

    def get_features(self, pixel_values, image_grid_thw=None, **kwargs):
        request = self.requests.get(pixel_values.data_ptr())
        if request is None or kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            return self.original(pixel_values, image_grid_thw, **kwargs)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
        keys, grids = request
        found, missing, patches, shapes = {}, [], [], []
        offset = 0
        for key, shape in zip(keys, grids):
            n = shape[0] * shape[1] * shape[2]
            if key in found:
                self.hits += 1
            elif key in self.entries:
                self.hits += 1
                found[key] = self.entries[key]
                self.entries.move_to_end(key)
            else:
                self.misses += 1
                found[key] = None
                missing.append(key)
                patches.append(pixel_values[offset:offset + n])
                shapes.append(shape)
            offset += n
        if missing:
            call_kw = {**kwargs, "return_dict": True}
            output = self.original(torch.cat(patches), image_grid_thw.new_tensor(shapes), **call_kw)
            lengths = [s[0] * s[1] * s[2] // self.merge for s in shapes]
            deep = [torch.split(t, lengths) for t in output.deepstack_features]
            for i, key in enumerate(missing):
                tensors = (output.pooler_output[i], *(level[i] for level in deep))
                value = tuple(t.detach().cpu().clone() for t in tensors)
                found[key] = value
                size = sum(t.numel() * t.element_size() for t in value)
                if size <= self.capacity:
                    self.entries[key] = value
                    self.bytes += size
                    while self.bytes > self.capacity:
                        _, old = self.entries.popitem(last=False)
                        self.bytes -= sum(t.numel() * t.element_size() for t in old)
        rows = [found[key] for key in keys]
        levels = [torch.cat([row[i] for row in rows]).to(pixel_values.device) for i in range(len(rows[0]))]
        lengths = [s[0] * s[1] * s[2] // self.merge for s in grids]
        return BaseModelOutputWithDeepstackFeatures(pooler_output=torch.split(levels[0], lengths),
                                                   deepstack_features=levels[1:])
