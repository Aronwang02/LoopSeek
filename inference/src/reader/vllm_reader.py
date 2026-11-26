"""Frozen reader in an isolated vLLM environment; training keeps its own Torch.

Images/video frames go directly to vLLM for multimodal preprocessing and inference.
The local processor supplies the chat template and tokenizer only.
"""
import atexit
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Connection

import numpy as np
import torch

from .qwen_reader import QwenReader, _to_pil, configure_processor


class VLLMReader(QwenReader):
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoConfig, AutoProcessor, GenerationConfig
        from vllm import LLM
        self.cfg = rc = cfg["reader"]
        self.model_key = rc.get("model", "reader")
        path = cfg["models"][self.model_key]["path"]
        model_cfg = AutoConfig.from_pretrained(path)
        if model_cfg.model_type != 'qwen3_vl':
            raise ValueError('the isolated vLLM reader currently supports Qwen3-VL checkpoints')
        self.visual_token_ids = {model_cfg.image_token_id, model_cfg.video_token_id}
        if rc.get('sdpa_backend'):
            raise ValueError('reader.sdpa_backend requires the HF engine; vLLM uses its own kernels')
        self.processor = AutoProcessor.from_pretrained(path)
        self.token_px = configure_processor(self.processor, rc)
        self.device = torch.device("cpu")        # local token handling; vLLM owns the GPU model
        self.vision_cache = None
        self.generation_config = GenerationConfig.from_pretrained(path)
        vc = rc.get("vllm", {})
        self.model = LLM(model=path, dtype=rc["dtype"], tensor_parallel_size=1,
                         max_model_len=int(vc.get("max_model_len", 32768)),
                         max_num_seqs=int(vc.get("max_num_seqs") or rc["batch_size"]),
                         gpu_memory_utilization=float(vc.get("gpu_memory_utilization", .55)),
                         enforce_eager=bool(vc.get("enforce_eager", False)),
                         max_num_batched_tokens=int(vc.get("max_num_batched_tokens", 8192)),
                         limit_mm_per_prompt={"image": 256, "video": 1},
                         skip_mm_profiling=True, logprobs_mode="raw_logprobs",
                         enable_prefix_caching=True, seed=cfg["project"]["seed"])
        self.stats = {"calls": 0, "seconds": 0., "prompt_tokens": 0, "visual_tokens": 0, "new_tokens": 0}

    def _request_batches(self, items):
        # Queue all ready requests. vLLM admits work as slots free up, without waiting
        # for the slowest request in an artificial HF-sized batch to finish first.
        if items:
            yield items

    def _generate(self, message_batch, max_new_tokens, letters_batch=None):
        from vllm import SamplingParams
        started = time.perf_counter()
        prompts, params = [], []
        eos = self.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        for i, messages in enumerate(message_batch):
            images, videos = [], []
            for message in messages:
                for c in message["content"]:
                    if c["type"] == "image":
                        images.append(_to_pil(c, self.token_px))
                    elif c["type"] == "video":
                        videos.append((np.stack(c["video"]), {"fps": c["fps"], "total_num_frames": len(c["video"]),
                                       "frames_indices": [t * c["fps"] for t in c["times"]],
                                       "do_sample_frames": False}))
            prompt = {"prompt": self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
            media = {**({"image": images} if images else {}), **({"video": videos} if videos else {})}
            if media:
                prompt["multi_modal_data"] = media
                kw = {"do_sample_frames": False} if videos else {}
                if images and (rc_tokens := self.cfg.get("frame_visual_tokens")):
                    kw.update(max_pixels=int(rc_tokens * self.token_px ** 2),
                              min_pixels=int(self.cfg["frame_min_visual_tokens"] * self.token_px ** 2))
                prompt["mm_processor_kwargs"] = kw
            prompts.append(prompt)
            ids = [self._letter_id(x) for x in letters_batch[i]] if letters_batch is not None else None
            params.append(SamplingParams(temperature=0., top_p=1., top_k=-1, max_tokens=max_new_tokens,
                                          repetition_penalty=self.generation_config.repetition_penalty,
                                          stop_token_ids=eos, logprob_token_ids=ids or None))
        t0 = time.perf_counter()
        outputs = self.model.generate(prompts, params, use_tqdm=False)
        generated = time.perf_counter() - t0
        texts, metas = [], []
        for i, output in enumerate(outputs):
            prompt_tokens = len(output.prompt_token_ids)
            visual_tokens = sum(token in self.visual_token_ids for token in output.prompt_token_ids)
            ct, vt = self.cfg.get("context_tokens"), self.cfg.get("visual_tokens")
            if (ct and prompt_tokens > ct) or (vt and visual_tokens > vt):
                raise ValueError(f"reader budget exceeded: prompt={prompt_tokens} visual={visual_tokens}")
            result = output.outputs[0]
            texts.append(self.processor.tokenizer.decode(result.token_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False))
            meta = {"prompt_tokens": prompt_tokens, "visual_tokens": visual_tokens, "new_tokens": len(result.token_ids),
                    "generation_seconds": generated, "prepare_seconds": t0 - started,
                    "batch_size": len(message_batch), "engine": "vllm"}
            if letters_batch is not None:
                letters = letters_batch[i]
                values = [result.logprobs[0][self._letter_id(x)].logprob for x in letters]
                meta["letter_probs"] = dict(zip(letters, torch.tensor(values).softmax(-1).tolist()))
            metas.append(meta)
        elapsed = time.perf_counter() - started
        for meta in metas:
            meta["batch_seconds"] = elapsed
        self.stats["calls"] += len(outputs)
        self.stats["seconds"] += elapsed
        self.stats["prompt_tokens"] += sum(m["prompt_tokens"] for m in metas)
        self.stats["visual_tokens"] += sum(m["visual_tokens"] for m in metas)
        self.stats["new_tokens"] += sum(m["new_tokens"] for m in metas)
        return texts, metas

    def letter_probs(self, message_batch, letters_batch):
        _, meta = self._generate(message_batch, 1, letters_batch)
        return [m["letter_probs"] for m in meta]


class ReaderProcess(QwenReader):
    """Private local pipe to an engine's Python; only CPU media and results cross it."""
    def __init__(self, cfg, device="cuda"):
        self.cfg, self.device = cfg["reader"], torch.device(device)
        self.model_key = self.cfg.get("model", "reader")
        python = self.cfg.get("vllm", {}).get("python") or os.environ.get("VLLM_PYTHON")
        if not python:
            root = Path(__file__).resolve().parents[2]
            python = next((str(p) for p in (Path('/opt/onejev-vllm/bin/python'), root / '.venv-vllm/bin/python')
                           if p.is_file() and os.access(p, os.X_OK)), None)
        if not python:
            raise ValueError("run bash scripts/setup_vllm_env.sh or set VLLM_PYTHON to an isolated vLLM environment")
        parent, child = socket.socketpair()
        env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn")
        # Reader generation is greedy. Avoid an unused FlashInfer sampling JIT
        # during engine warmup on machines without a CUDA compiler.
        env.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        visible = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        index = self.device.index or 0
        env["CUDA_VISIBLE_DEVICES"] = visible[index] if visible[0] else str(index)
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root
        self.proc = subprocess.Popen([python, "-m", "src.reader.vllm_reader", str(child.fileno())],
                                     pass_fds=(child.fileno(),), env=env, cwd=root)
        child.close()
        self.conn = Connection(parent.detach())
        self.closed = False
        atexit.register(self.close)
        try:
            self.conn.send(cfg)
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        status, result, stats = self.conn.recv()
        if status != "ok":
            raise RuntimeError("vLLM reader process failed:\n" + result)
        self.stats = stats
        return result

    def _call(self, method, *args):
        self.conn.send((method, args))
        return self._receive()

    def run(self, jobs):
        return self._call("run", jobs)

    def generate(self, messages, max_new_tokens):
        return self._call("generate", messages, max_new_tokens)

    def answer_letter_probs(self, jobs):
        return self._call("answer_letter_probs", jobs)

    def letter_probs(self, messages, letters):
        return self._call("letter_probs", messages, letters)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.conn.close()


def serve(fd):
    conn = Connection(fd)
    try:
        reader = VLLMReader(conn.recv())
        conn.send(("ok", None, reader.stats))
        while (request := conn.recv()) is not None:
            method, args = request
            try:
                result = getattr(reader, method)(*args)
                conn.send(("ok", result, reader.stats))
            except Exception:
                conn.send(("error", traceback.format_exc(), reader.stats))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        conn.send(("error", traceback.format_exc(), {}))
    finally:
        conn.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
