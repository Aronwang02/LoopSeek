from types import SimpleNamespace

import torch

from src.reader.vision_cache import VisionFeatureCache


def test_cache_preserves_image_video_order_and_all_deepstack_levels():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
    encoded = []
    def original(pixels, grids, **kwargs):
        encoded.append(len(grids))
        lengths = (grids.prod(-1) // 4).tolist()
        value = pixels.reshape(-1, 4, 3).mean(1)
        return BaseModelOutputWithDeepstackFeatures(pooler_output=value.split(lengths),
                                                    deepstack_features=[value * 2, value + 3])
    model = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2), get_image_features=original)
    cache = VisionFeatureCache(model, 1)
    a, b = torch.arange(12.).reshape(4, 3), torch.arange(24.).reshape(8, 3)
    pixels, grid = torch.cat([a, b, a]), torch.tensor([[1, 2, 2], [2, 2, 2], [1, 2, 2]])
    expected = original(pixels, grid)
    encoded.clear()
    for prefix in ('image', 'video'):
        names = ('pixel_values', 'image_grid_thw') if prefix == 'image' else ('pixel_values_videos', 'video_grid_thw')
        data = dict(zip(names, (pixels, grid)))
        cache.register(data, data)
        got = model.get_image_features(pixels, grid)
        for x, y in zip(got.pooler_output, expected.pooler_output):
            torch.testing.assert_close(x, y)
        for x, y in zip(got.deepstack_features, expected.deepstack_features):
            torch.testing.assert_close(x, y)
    assert encoded == [2]  # duplicate image and repeated video inputs never re-encode
    assert cache.hits == 4 and cache.misses == 2
    pixels = pixels.clone()
    pixels[0, 0] += 1
    data = dict(zip(('pixel_values', 'image_grid_thw'), (pixels, grid)))
    cache.register(data, data)
    got = model.get_image_features(pixels, grid)
    assert encoded == [2, 1]  # a changed pixel must invalidate the relevant entry
    assert not torch.equal(got.pooler_output[0], expected.pooler_output[0])


def test_cache_bound_does_not_evict_inflight_results():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
    def original(pixels, grids, **kwargs):
        value = pixels.reshape(-1, 4, 3).mean(1)
        return BaseModelOutputWithDeepstackFeatures(pooler_output=value.split(1), deepstack_features=[value * 2])
    model = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2), get_image_features=original)
    cache = VisionFeatureCache(model, 30 / 1024 ** 2)  # room for one entry only
    pixels, grid = torch.arange(36.).reshape(12, 3), torch.tensor([[1, 2, 2]] * 3)
    data = dict(pixel_values=pixels, image_grid_thw=grid)
    cache.register(data, data)
    got = model.get_image_features(pixels, grid)
    torch.testing.assert_close(torch.cat(got.pooler_output), pixels.reshape(-1, 4, 3).mean(1))
    assert len(cache.entries) == 1 and cache.bytes <= cache.capacity
