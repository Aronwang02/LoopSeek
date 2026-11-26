"""Download pinned model snapshots into $HF_HOME and record resolved paths (configs/model_paths.json).

Training needs: python scripts/download_models.py onejev_4b reader_qwen3vl8b nanojev_tokenizer
(onejev_4b = the backbone of System I; reader_qwen3vl8b = System II, the frozen answerer; nanojev_tokenizer = CPU tests
only; scripts/setup_env.sh downloads these three). Needs network access.
"""
import json
import os
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    # name: (repo, revision, allow_patterns)
    "reader": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5",
               ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
    "siglip2": ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                ["*.json", "*.safetensors", "*.model", "*.txt"]),
    "nanojev": ("C-Tianyu/NanoJev", "unified-games-v1",
                ["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*",
                 "README.md", "TRAINING_RECIPE.md", "SHA256_MANIFEST.json"]),
    "nanojev_tokenizer": ("C-Tianyu/NanoJev", "unified-games-v1", ["tokenizer/*"]),   # CPU tests only (tests/conftest.py)
    "qwen3_0p6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
                   ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
    "unitime_lora": ("zeqianli/UniTime", "4fdb258c8c6c8e30415746b725f88ac45114bfc1", None),
    "qwen2vl_7b": ("Qwen/Qwen2-VL-7B-Instruct", "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
                   ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
    "reader_qwen3vl8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                         ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
    "onejev_4b": ("OmniJev/OneJev-4B", "c88e18653ceb7a8770716287f55fdefc79d6b588",       # planned policy init
                  ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                  ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
    "qwen35_9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",       # backbone-size comparison
                  ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
}

RETRY_STATUS = (429, 500, 502, 503, 504)


def download(repo, rev, patterns, attempts=8):
    """snapshot_download, retried with backoff (30 s doubling to 10 min, ~35 min in all) while the Hub answers 429 / 5xx
    or the connection fails (a launch once failed on '429 Too Many Requests ... maximum queue size reached' listing a
    pinned revision). Other HTTP errors (404, 401) are raised at once."""
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(repo, revision=rev, allow_patterns=patterns, max_workers=8)
        except (OSError, httpx.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt == attempts or (status is not None and status not in RETRY_STATUS):
                raise
            wait = min(30 * 2 ** (attempt - 1), 600)
            print(json.dumps({"hub_retry": repo, "status": status, "attempt": attempt, "wait_s": wait}), flush=True)
            time.sleep(wait)


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    out = ROOT / "configs" / "model_paths.json"
    record = json.loads(out.read_text()) if out.exists() else {}
    for name in names:
        repo, rev, patterns = MODELS[name]
        path = download(repo, rev, patterns)
        record[name] = {"repo": repo, "revision": rev, "path": path}
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({name: record[name]}), flush=True)
    print("HF_HOME", os.environ.get("HF_HOME"))
