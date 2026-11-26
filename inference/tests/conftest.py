"""CPU fixtures: tiny random Qwen3 backbone with the real NanoJev tokenizer, a stub frozen
encoder, and a synthetic variable-frame-rate video with known timestamps."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)   # the login node has 128 cores shared with others; default threading stalls

from src.common.config import load_config  # noqa: E402
from src.policy.model import JevPolicy  # noqa: E402
from src.video.assets import FeatureStore, VideoAsset  # noqa: E402

VFR_PTS_MS = [0, 40, 80, 120, 400, 440, 480, 1500, 1540, 1580] + [1600 + 50 * i for i in range(90)]


def frame_color(i):
    # unique for i < 100, 25 levels apart per channel so compression cannot confuse frames
    return np.array([(i % 10) * 25, (i // 10) * 25, 128], dtype=np.uint8)


@pytest.fixture(scope="session")
def cfg():
    c = load_config([ROOT / "configs" / "default.yaml"])
    c["budget"]["min_window_sec"] = 0.2      # the synthetic video is only 6.1 s long
    return c


@pytest.fixture(scope="session")
def vfr_video(tmp_path_factory):
    """MPEG-4 file whose frames have irregular pts (ms); frame i is a solid color."""
    path = tmp_path_factory.mktemp("vid") / "vfr.mp4"
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=25)
        st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
        st.time_base = Fraction(1, 1000)
        st.codec_context.time_base = Fraction(1, 1000)
        st.codec_context.gop_size = 8
        st.codec_context.max_b_frames = 0
        for i, p in enumerate(VFR_PTS_MS):
            img = np.tile(frame_color(i), (48, 64, 1))
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            fr.pts = p
            fr.time_base = Fraction(1, 1000)
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return path


class StubEncoder:
    """Deterministic 'frozen encoder': 4 tokens x 768 per frame from the frame's quadrant colors."""
    tag = "stub"

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(3, 768, generator=g)
        self.calls = 0
        self.encode_seconds = 0.0

    def encode(self, frames):
        self.calls += len(frames)
        out = []
        for f in frames:
            h, w = f.shape[0] // 2, f.shape[1] // 2
            quads = [f[:h, :w], f[:h, w:], f[h:, :w], f[h:, w:]]
            c = torch.tensor(np.stack([q.reshape(-1, 3).mean(0) for q in quads]) / 255.0, dtype=torch.float32)
            out.append(c @ self.w)
        return torch.stack(out)


@pytest.fixture()
def features(tmp_path):
    return FeatureStore(tmp_path / "feats", "stub")


@pytest.fixture()
def asset(vfr_video, tmp_path):
    return VideoAsset(vfr_video, tmp_path / "cache", max_side=None)


def nanojev_tokenizer_dir(cfg):
    """NanoJev tokenizer for the tiny CPU models: from the full NanoJev download (older 0.6B experiments) or, for the
    4B pipeline, `python scripts/download_models.py nanojev_tokenizer` (11 MB); None if neither is present."""
    cands = [cfg["models"]["nanojev"].get("path")]
    rec = ROOT / "configs" / "model_paths.json"
    if rec.exists():
        cands.append(json.loads(rec.read_text()).get("nanojev_tokenizer", {}).get("path"))
    return next((f"{c}/tokenizer" for c in cands if c and Path(f"{c}/tokenizer").is_dir()), None)


@pytest.fixture(scope="session")
def tokenizer(cfg):
    from transformers import AutoTokenizer
    d = nanojev_tokenizer_dir(cfg)
    if d is None:
        pytest.skip("NanoJev tokenizer missing: python scripts/download_models.py nanojev_tokenizer")
    return AutoTokenizer.from_pretrained(d)


def tiny_backbone(vocab):
    from transformers import Qwen3Config, Qwen3Model
    conf = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       max_position_embeddings=8192, use_cache=False)
    torch.manual_seed(0)
    return Qwen3Model(conf).float()


@pytest.fixture()
def tiny_policy(tokenizer, features, cfg):
    from src.policy.lora import apply_lora
    pol = JevPolicy(tiny_backbone(len(tokenizer)), tokenizer, vis_dim=768, tokens_per_frame=4)
    lc = cfg["policy"]["lora"]
    pol.lora_modules = apply_lora(pol.backbone, lc["targets"], lc["r"], lc["alpha"], lc["dropout"])
    torch.manual_seed(1)
    for n, p in pol.named_parameters():    # non-zero LoRA B so tests exercise the LoRA path too
        if n.endswith("lora_B"):
            p.data.normal_(0, 0.02)
    pol.feature_store = features
    return pol


class StubReader:
    """Frozen-answerer stand-in: always answers B; records its jobs."""

    def __init__(self):
        self.jobs = []

    def run(self, jobs):
        self.jobs.extend(jobs)
        meta = {"batch_seconds": 0.0, "batch_size": len(jobs), "prompt_tokens": 1, "visual_tokens": 0, "new_tokens": 1}
        return [{"text": "B", "parsed": "B", "meta": meta} for _ in jobs]


class LabelReader(StubReader):
    """StubReader that also returns option-letter probabilities (verifier labelling, budget.label_pE)."""

    def answer_letter_probs(self, jobs):
        return [{"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1} for _ in jobs]
