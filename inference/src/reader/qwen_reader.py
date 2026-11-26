"""Frozen Qwen-VL reader / final answerer: Qwen2.5-VL-7B or Qwen3-VL-8B (cfg reader.model).

One model and one set of prompts/decoding parameters serve every system (B0-B4). Frames are
given as individual images, each preceded by its real timestamp. Greedy decoding (each model's own
generation_config otherwise, e.g. repetition_penalty 1.05 for Qwen2.5-VL, 1.0 for Qwen3-VL). Frames get the
model's official preprocessing unless reader.frame_visual_tokens caps them (converted with the model's
pixels per token, patch x merge).
"""
import re
import time

import numpy as np
import torch
from PIL import Image

from ..evaluation.scoring import parse_answer

OBS_INSTRUCTION = (
    "Describe what these frames show that is relevant to the question, and mention timestamps. "
    "Only report what is visible in these frames. If they do not contain enough evidence to "
    "answer, start with \"Insufficient evidence.\" and then state what is visible.")
ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."
# reader.answer_prompt = r1_cot (2026-10-04, protocol check against Video-MTR): the Video-R1 eval prompt that Video-MTR
# copies (Xyuan13/Video-MTR vagen/env/video/prompt.py SYSTEM_PROMPT_VIDEO_R1_EVAL, QUESTION_TEMPLATE, TYPE_TEMPLATE
# "multiple choice"), scored strictly: the stripped <answer> content must equal an option letter (their reward.py).
R1_SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>")
R1_INSTRUCTION = (
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', "
    "'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between "
    "the <answer> and </answer> tags."
    " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.")
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_answer_tag(text, letters):
    """Strict r1_cot scoring: the first <answer>...</answer> content, stripped, must be exactly an option letter."""
    m = _ANSWER_TAG.search(text or "")
    if not m:
        return None
    s = m.group(1).strip()
    return s if s in set(letters) else None
# reader.obs_prompt = selfassess3: a three-level self-assessment first (graded input for the verifier and the
# controller; the option letter is named only when the frames alone determine it), then the same description.
OBS_INSTRUCTION_SELFASSESS3 = (
    "Describe what these frames show that is relevant to the question, and mention timestamps. "
    "Only report what is visible in these frames; do not guess. Start your reply with exactly one of: "
    "\"Sufficient evidence for (X).\" where X is the option letter, if these frames alone determine the answer; "
    "\"Partial evidence.\" if they show content related to the question but do not determine the answer; "
    "\"Insufficient evidence.\" if they show nothing related to the question. Then state what is visible.")
OBS_INSTRUCTIONS = {"v1": OBS_INSTRUCTION, "selfassess3": OBS_INSTRUCTION_SELFASSESS3}


def _options_text(options):
    return "\n".join(f"({k}) {v}" for k, v in options.items())


def observe_messages(frames, times, question, options, source, interval, duration, instruction=OBS_INSTRUCTION):
    content = []
    if source == "clip":
        content.append({"type": "text", "text": f"These frames are sampled in temporal order from "
                        f"{interval[0]:.1f}s to {interval[1]:.1f}s of a {duration:.1f}s video."})
    else:
        content.append({"type": "text", "text": f"These frames are selected from a {duration:.1f}s video."})
    for f, t in zip(frames, times):
        content.append({"type": "text", "text": f"\nFrame at {t:.1f}s:"})
        content.append({"type": "image", "image": f})
    content.append({"type": "text", "text": f"\nQuestion: {question}\nOptions:\n{_options_text(options)}\n"
                    f"{instruction}"})
    return [{"role": "user", "content": content}]


def answer_messages(frames, times, question, options, notes, duration, style="direct", input_mode="image", intro=True,
                    video_fps="real"):
    """input_mode=video (2026-10-04, literature protocol check): the frames go in as ONE video (time-ordered, the
    processor merges consecutive pairs) with fps = n / duration and no per-frame timestamps; image (default): each frame
    an image preceded by its real timestamp."""
    # intro=False / video_fps=2.0 (2026-10-04, LENS protocol check): no duration sentence; fps 2.0 reproduces
    # transformers 4.51's default when no fps is passed (1 s per merged pair whatever the real spacing)
    content = [{"type": "text", "text": f"You are given frames from a {duration:.1f}s video"
                + (" and notes from earlier inspection of the video." if notes else ".")}] if intro else []
    if not frames:
        content.append({"type": "text", "text": "\n(No frames were inspected.)"})
    if input_mode == "video" and frames:
        order = sorted(range(len(frames)), key=lambda i: times[i])
        if intro:
            content.append({"type": "text", "text": "\n"})
        fps = len(frames) / duration if video_fps == "real" else float(video_fps)
        content.append({"type": "video", "video": [frames[i] for i in order], "fps": fps,
                        "times": [float(times[i]) for i in order]})
    else:
        for f, t in zip(frames, times):
            content.append({"type": "text", "text": f"\nFrame at {t:.1f}s:"})
            content.append({"type": "image", "image": f})
    if notes:
        content.append({"type": "text", "text": "\nNotes:\n" + "\n".join(f"- [{w}] {x}" for w, x in notes)})
    content.append({"type": "text", "text": ("\n" if intro else "") + f"Question: {question}\nOptions:\n{_options_text(options)}\n"
                    f"{R1_INSTRUCTION if style == 'r1_cot' else ANSWER_INSTRUCTION}"})
    user = {"role": "user", "content": content}
    if style == "r1_cot":
        return [{"role": "system", "content": [{"type": "text", "text": R1_SYSTEM}]}, user]
    return [user]


def _to_pil(c, token_px=28):
    img = Image.fromarray(c["image"])
    mp = c.get("max_pixels")
    if mp and img.width * img.height > mp:   # pre-shrink; the processor's token_px rounding follows
        s = (mp / (img.width * img.height)) ** 0.5
        img = img.resize((max(token_px, int(img.width * s)), max(token_px, int(img.height * s))), Image.BICUBIC)
    return img


def apply_reader_backend(cfg):
    """Call right after load_config, before any model is built: reader.sdpa_backend=math restricts PyTorch SDPA to
    the math kernel for the whole process (the only cross-process bit-reproducible setting found, 2026-09-30)."""
    if cfg["reader"].get("sdpa_backend") == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return True
    return False


def configure_processor(processor, rc):
    """Left padding; per-frame pixel limits only when reader.frame_visual_tokens is set (else the model's
    official preprocessing stays). Returns pixels per token side."""
    processor.tokenizer.padding_side = "left"
    ip = processor.image_processor
    token_px = ip.patch_size * ip.merge_size       # 28 (Qwen2.5-VL: 14 x 2), 32 (Qwen3-VL: 16 x 2)
    if rc.get("frame_visual_tokens"):
        ip.size = {"shortest_edge": rc["frame_min_visual_tokens"] * token_px ** 2,
                   "longest_edge": rc["frame_visual_tokens"] * token_px ** 2}
    return token_px


def build_reader(cfg, device="cuda"):
    engine = cfg["reader"].get("engine", "hf")
    if engine == "hf":
        return QwenReader(cfg, device)
    if engine == "vllm":
        from .vllm_reader import ReaderProcess
        return ReaderProcess(cfg, device)
    raise ValueError(f"unknown reader.engine: {engine}")


class QwenReader:
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        rc = cfg["reader"]
        self.model_key = rc.get("model", "reader")
        path = cfg["models"][self.model_key]["path"]
        self.cfg = rc
        self.device = torch.device(device)
        # 2026-09-30: reader.sdpa_backend=math -> PyTorch SDPA restricted to the math kernel for the WHOLE process,
        # set before the model is built. Verified bit-reproducible across processes (50/50);
        # switching the flags only around each call was not reproducible (29/50 even in-process), so this is a
        # process-level setting: a run uses either default sdpa or math, never both. Eager is not used as a reference
        # (its batched bf16 generations are corrupted, runs/p24_q25).
        if rc.get("sdpa_backend") == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=getattr(torch, rc["dtype"]), attn_implementation=rc["attn_implementation"],
            device_map={"": self.device})
        self.model.eval().requires_grad_(False)
        self.sdpa_backend = rc.get("sdpa_backend")
        assert self.sdpa_backend in (None, "math"), self.sdpa_backend
        assert self.sdpa_backend is None or rc["attn_implementation"] == "sdpa"
        self.processor = AutoProcessor.from_pretrained(path)
        self.token_px = configure_processor(self.processor, rc)
        self.stats = {"calls": 0, "seconds": 0.0, "prompt_tokens": 0, "visual_tokens": 0, "new_tokens": 0}
        self.vision_cache = None
        if rc.get("vision_cache_mb", 0) and self.model.config.model_type == "qwen3_vl":
            from .vision_cache import VisionFeatureCache
            self.vision_cache = VisionFeatureCache(self.model.model, rc["vision_cache_mb"])

    def answer_style(self):
        """reader.answer_prompt: direct (default, letter only) | r1_cot (Video-R1 / Video-MTR CoT, strict tag parse)."""
        return self.cfg.get("answer_prompt", "direct")

    def obs_prompt(self):
        """reader.obs_prompt is a name (v1 | selfassess3) or a map reader-model key -> name, so every system of
        one reader group uses the same observation prompt (2026-09-30: Qwen2.5-VL selfassess3, Qwen3-VL v1)."""
        op = self.cfg.get("obs_prompt", "v1")
        return op[self.model_key] if isinstance(op, dict) else op

    def _prepare(self, message_batch):
        proc = self.processor
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in message_batch]
        images = [_to_pil(c, self.token_px) for m in message_batch for msg in m for c in msg["content"] if c["type"] == "image"]
        vids = [c for m in message_batch for msg in m for c in msg["content"] if c["type"] == "video"]
        kw = {}
        if vids:   # frames of one video share the decode size; fps -> second_per_grid_ts = 2 / fps
            # frames_indices = time x fps (2026-10-07): Qwen3-VL writes "<t seconds>" before each merged frame pair from
            # frames_indices / fps (None crashed); with these indices t is each frame's real time, as qwen-vl-utils
            # gives it. Qwen2.5-VL does not read frames_indices.
            kw = {"videos": [np.stack(c["video"]) for c in vids],
                  "videos_kwargs": {"video_metadata": [{"fps": c["fps"], "total_num_frames": len(c["video"]),
                                                        "frames_indices": [t * c["fps"] for t in c["times"]]}
                                                       for c in vids], "do_sample_frames": False}}
        inputs = proc(text=texts, images=images or None, padding=True, return_tensors="pt", **kw)
        n_prompt = inputs["attention_mask"].sum(-1).tolist()
        merge = proc.image_processor.merge_size ** 2
        per_img = (inputs["image_grid_thw"].prod(-1) // merge).tolist() if "image_grid_thw" in inputs else []
        per_vid = (inputs["video_grid_thw"].prod(-1) // merge).tolist() if "video_grid_thw" in inputs else []
        vis = []
        k = kv = 0
        for m in message_batch:
            n_img = sum(1 for msg in m for c in msg["content"] if c["type"] == "image")
            n_vid = sum(1 for msg in m for c in msg["content"] if c["type"] == "video")
            vis.append(int(sum(per_img[k:k + n_img]) + sum(per_vid[kv:kv + n_vid])))
            k += n_img
            kv += n_vid
        ct, vt = self.cfg.get("context_tokens"), self.cfg.get("visual_tokens")    # null = no limit
        for p, v in zip(n_prompt, vis):
            if (ct and p > ct) or (vt and v > vt):
                raise ValueError(f"reader budget exceeded: prompt={p} visual={v}")
        # BatchFeature.to mutates the object: retain CPU values for content keys.
        cpu_inputs = dict(inputs)
        inputs = inputs.to(self.device)
        if getattr(self, "vision_cache", None) is not None:
            self.vision_cache.register(cpu_inputs, inputs)
        return inputs, n_prompt, vis

    def _letter_id(self, letter):
        if not hasattr(self, "_letter_ids"):
            self._letter_ids = {}
        if letter not in self._letter_ids:
            ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, (letter, ids)
            self._letter_ids[letter] = ids[0]
        return self._letter_ids[letter]

    def _attn(self):
        import contextlib
        return contextlib.nullcontext()

    def letter_probs(self, message_batch, letters_batch):
        """Next-token probabilities at the first answer position of the same prompt the answerer generates
        from, renormalized over the option letters only. One forward (last position only), no generation.
        Used for the unparsed-answer fallback and for verifier labels (p_E = probability of the correct letter)."""
        t0 = time.perf_counter()
        inputs, n_prompt, vis = self._prepare(message_batch)
        with torch.inference_mode(), self._attn():
            out = self.model(**inputs, logits_to_keep=1, use_cache=False)
        logits = out.logits[:, -1, :].float()
        res = []
        for row, letters in zip(logits, letters_batch):
            letters = list(letters)
            p = row[[self._letter_id(x) for x in letters]].softmax(-1).tolist()
            res.append(dict(zip(letters, p)))
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["calls"] += len(message_batch)
        self.stats["prompt_tokens"] += sum(n_prompt)
        self.stats["visual_tokens"] += sum(vis)
        return res

    def _request_batches(self, items):
        """HF executes fixed batches; serving engines can override submission grouping."""
        bs = self.cfg["batch_size"]
        for start in range(0, len(items), bs):
            yield items[start:start + bs]

    def answer_letter_probs(self, jobs):
        """Letter probabilities for answer jobs (same messages as run(kind=answer)), batched."""
        out = []
        for chunk in self._request_batches(jobs):
            msgs = [answer_messages(j["frames"], j["times"], j["question"], j["options"], j["notes"], j["duration"],
                                    self.answer_style(), self.cfg.get("answer_input", "image"),
                                    self.cfg.get("answer_intro", True), self.cfg.get("video_fps", "real")) for j in chunk]
            out += self.letter_probs(msgs, [list(j["options"].keys()) for j in chunk])
        return out

    def _generate(self, message_batch, max_new_tokens, letters_batch=None):
        proc = self.processor
        started = time.perf_counter()
        inputs, n_prompt, vis = self._prepare(message_batch)
        t0 = time.perf_counter()
        # Capture raw first-position logits before generate's logits processors.
        # Keep only option logits, not a vocabulary-sized tensor for every token.
        first = []
        ids = [[self._letter_id(x) for x in ls] for ls in letters_batch] if letters_batch is not None else None
        def capture(module, args, output):
            if not first:
                first.extend(output.logits[i, -1, ix].detach().float().softmax(-1) for i, ix in enumerate(ids))
        hook = self.model.register_forward_hook(capture) if ids is not None else None
        try:
            with torch.inference_mode(), self._attn():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None)
        finally:
            if hook is not None:
                hook.remove()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_secs = time.perf_counter() - t0
        new = out[:, inputs["input_ids"].shape[1]:]
        texts_out = proc.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        n_new = (new != proc.tokenizer.pad_token_id).sum(-1).tolist()
        letter_probs = [dict(zip(ls, p.tolist())) for ls, p in zip(letters_batch, first)] if ids is not None else None
        if ids is not None and len(first) != len(message_batch):
            raise RuntimeError("generate did not expose first-token logits")
        secs = time.perf_counter() - started
        self.stats["calls"] += len(message_batch)
        self.stats["seconds"] += secs
        self.stats["prompt_tokens"] += sum(n_prompt)
        self.stats["visual_tokens"] += sum(vis)
        self.stats["new_tokens"] += sum(n_new)
        meta = [{"prompt_tokens": p, "visual_tokens": v, "new_tokens": n, "batch_seconds": secs,
                 "generation_seconds": generation_secs, "prepare_seconds": t0 - started,
                 "batch_size": len(message_batch)} for p, v, n in zip(n_prompt, vis, n_new)]
        if letter_probs is not None:
            for m, pr in zip(meta, letter_probs):
                m["letter_probs"] = pr
        return texts_out, meta

    def generate(self, message_batch, max_new_tokens):
        """Generic greedy generation (planner, frame scorer) with the same frozen checkpoint."""
        out_t, out_m = [], []
        for chunk in self._request_batches(message_batch):
            t, m = self._generate(chunk, max_new_tokens)
            out_t += t
            out_m += m
        return out_t, out_m

    def run(self, jobs):
        """jobs: list of dicts {kind: observe|answer, frames, times, question, options, duration,
        source, interval, notes}. Returns list of dicts {text, parsed?, meta}. Batched by kind."""
        results = [None] * len(jobs)
        for kind in ("observe", "answer"):
            idx = [i for i, j in enumerate(jobs) if j["kind"] == kind]
            for chunk in self._request_batches(idx):
                if kind == "observe":
                    msgs = [observe_messages(jobs[i]["frames"], jobs[i]["times"], jobs[i]["question"],
                                             jobs[i]["options"], jobs[i]["source"], jobs[i]["interval"],
                                             jobs[i]["duration"],
                                             OBS_INSTRUCTIONS[self.obs_prompt()]) for i in chunk]
                    texts, meta = self._generate(msgs, self.cfg["observation_max_new_tokens"])
                else:
                    msgs = [answer_messages(jobs[i]["frames"], jobs[i]["times"], jobs[i]["question"],
                                            jobs[i]["options"], jobs[i]["notes"], jobs[i]["duration"],
                                            self.answer_style(), self.cfg.get("answer_input", "image"),
                                    self.cfg.get("answer_intro", True), self.cfg.get("video_fps", "real"))
                            for i in chunk]
                    texts, meta = self._generate(msgs, self.cfg["answer_max_new_tokens"],
                                                 [list(jobs[i]["options"]) for i in chunk])
                for i, t, m in zip(chunk, texts, meta):
                    r = {"text": t, "meta": m}
                    if kind == "answer":
                        r["letter_probs"] = m.get("letter_probs")
                        r["parsed"] = (parse_answer_tag(t, jobs[i]["options"].keys()) if self.answer_style() == "r1_cot"
                                       else parse_answer(t, jobs[i]["options"].keys()))
                        m["answer_source"] = "generated" if r["parsed"] is not None else "unparsed"
                    results[i] = r
                if kind == "answer" and self.cfg.get("answer_fallback", "none") == "letter_logits":
                    # 2026-09-29 rule for every system: no legal letter in the generated text (e.g. "None of the
                    # options...") -> the option letter with the highest next-token probability, same prompt
                    # caption queries (ActivityNet Captions grounding rows) have no options: nothing to fall back to
                    miss = [k for k, i in enumerate(chunk) if results[i]["parsed"] is None and jobs[i]["options"]]
                    if miss:
                        probs = [results[chunk[k]]["letter_probs"] for k in miss]
                        for k, pr in zip(miss, probs):
                            r = results[chunk[k]]
                            r["parsed"] = max(pr, key=pr.get)
                            r["letter_probs"] = pr
                            r["meta"]["answer_source"] = "letter_logits_fallback"
        return results
