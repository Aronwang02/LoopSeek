"""Reader adaptation (Qwen2.5-VL-7B and Qwen3-VL-8B), processor only (no model weights, CPU). By default each
reader keeps its official preprocessing (no per-frame or per-call cap); an optional per-frame cap in visual
tokens is converted with the model's pixels per token. A decode-size frame goes through the chat template and
the processor with as many image-pad tokens as the grid implies."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.deciders import QwenChoiceDecider
from src.reader.qwen_reader import _to_pil, answer_messages, configure_processor, observe_messages

# models key, pixels per token side, visual tokens of one 640x360 frame under the official preprocessing
READERS = [("reader", 28, 299), ("reader_qwen3vl8b", 32, 220)]


def processor(cfg, key):
    from transformers import AutoProcessor
    path = cfg["models"][key]["path"]
    if not path or not Path(path).exists():
        pytest.skip(f"{key} not downloaded")
    return AutoProcessor.from_pretrained(path)


@pytest.mark.parametrize("key,token_px,n_tokens", READERS)
def test_official_preprocessing_by_default(cfg, key, token_px, n_tokens):
    rc = cfg["reader"]
    assert rc["frame_visual_tokens"] is None and rc["visual_tokens"] is None and rc["context_tokens"] is None
    proc, official = processor(cfg, key), processor(cfg, key)
    assert configure_processor(proc, rc) == token_px
    assert proc.image_processor.size == official.image_processor.size            # untouched
    assert proc.tokenizer.padding_side == "left"
    assert QwenChoiceDecider(SimpleNamespace(cfg=rc, token_px=token_px)).max_pixels is None


@pytest.mark.parametrize("key,token_px,n_tokens", READERS)
def test_optional_frame_cap_in_tokens(cfg, key, token_px, n_tokens):
    rc = {**cfg["reader"], "frame_visual_tokens": 256, "decider_frame_visual_tokens": 128}
    proc = processor(cfg, key)
    configure_processor(proc, rc)
    assert proc.image_processor.size == {"shortest_edge": 4 * token_px ** 2, "longest_edge": 256 * token_px ** 2}
    assert QwenChoiceDecider(SimpleNamespace(cfg=rc, token_px=token_px)).max_pixels == 128 * token_px ** 2


@pytest.mark.parametrize("key,token_px,n_tokens", READERS)
def test_decode_size_frame_through_template(cfg, key, token_px, n_tokens):
    proc = processor(cfg, key)
    configure_processor(proc, cfg["reader"])
    rng = np.random.default_rng(0)
    frames = [(rng.random((360, 640, 3)) * 255).astype(np.uint8) for _ in range(2)]   # decode_max_side 640
    opts = {"A": "red", "B": "green", "C": "blue", "D": "gray"}
    for msgs in (observe_messages(frames, [1.0, 2.5], "What color?", opts, "select", None, 10.0),
                 answer_messages(frames, [1.0, 2.5], "What color?", opts, [(2, "saw green")], 10.0)):
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images = [_to_pil(c, token_px) for c in msgs[0]["content"] if c["type"] == "image"]
        inputs = proc(text=[text], images=images, return_tensors="pt")
        per_img = (inputs["image_grid_thw"].prod(-1) // proc.image_processor.merge_size ** 2).tolist()
        pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert int((inputs["input_ids"] == pad).sum()) == sum(per_img)
        assert per_img == [n_tokens, n_tokens]
        assert "Frame at 1.0s:" in text and "What color?" in text
