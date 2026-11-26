"""Run directories: resolved config, code state, package versions, Slurm/GPU info, wandb."""
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

from .config import dump_config
from .utils import git_state, write_json


def env_versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("torch", "transformers", "safetensors", "av", "numpy", "wandb"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return out


def setup_run(cfg, name, argv=None):
    """Creates runs/<name>/ with config.yaml and run_info.json. Resumes if it already exists."""
    run_dir = Path(cfg["paths"]["runs_root"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    info = {"name": name, "argv": argv or sys.argv, "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_state(),
            "env": env_versions(), "model_revisions": {k: v.get("revision") for k, v in cfg["models"].items()}}
    hist = run_dir / "run_info.jsonl"
    with open(hist, "a") as f:
        f.write(json.dumps(info) + "\n")
    write_json(run_dir / "run_info.json", info)
    return run_dir


def wandb_settings(cfg):
    """W&B target: environment (WANDB_*) over config (project.wandb_*)."""
    p = cfg["project"]
    return {"entity": os.environ.get("WANDB_ENTITY") or p.get("wandb_entity") or None,
            "project": os.environ.get("WANDB_PROJECT") or p["wandb_project"],
            "mode": os.environ.get("WANDB_MODE") or p["wandb_mode"],
            "group": os.environ.get("WANDB_RUN_GROUP") or None}


def flat_scalars(d, prefix=""):
    """Nested dict -> {"a/b": number}; bools become 0/1, non-numeric values are dropped."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat_scalars(v, key + "/"))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[key] = v
        elif isinstance(v, (list, tuple)) and 0 < len(v) <= 4 and all(isinstance(x, (int, float)) for x in v):
            out.update(flat_scalars({str(i): x for i, x in enumerate(v)}, key + "/"))
    return out


def init_wandb(cfg, run_dir, name, job_type):
    """One W&B run per (name, job_type); resumes the same run when a stage is rerun. Online mode that cannot reach
    the server falls back to offline (sync later with `wandb sync runs/<name>/wandb/offline-*`)."""
    w = wandb_settings(cfg)
    if w["mode"] == "disabled":
        return None
    import wandb
    tags = [job_type] + ([f"ans={os.environ['ANS']}"] if os.environ.get("ANS") else [])
    kw = dict(entity=w["entity"], project=w["project"], group=w["group"], name=name, job_type=job_type, tags=tags,
              dir=str(run_dir), config={**cfg, "argv": sys.argv, "run_dir": str(run_dir)}, resume="allow",
              id=re.sub(r"[^A-Za-z0-9_.-]", "-", f"{name}-{job_type}")[:120])
    try:
        return wandb.init(mode=w["mode"], **kw)
    except Exception as exc:                   # no network / not logged in: keep training, log offline
        if w["mode"] != "online":
            raise
        print(json.dumps({"wandb_online_failed": repr(exc)[:300], "fallback": "offline"}), flush=True)
        return wandb.init(mode="offline", **kw)


def wandb_log(wb, log, step=None):
    if wb is not None:
        wb.log(flat_scalars(log), step=step)


def wandb_summary(name, job_type, run_dir, summary, cfg=None):
    """For evaluation-type scripts: one short W&B run whose summary holds the scalar results (nested keys a/b)."""
    if cfg is None:
        from src.common.config import load_config
        cfg = load_config([str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")])
    wb = init_wandb(cfg, Path(run_dir), name, job_type)
    if wb is not None:
        wb.summary.update(flat_scalars(summary))
        wb.finish()


def _hold_pidfile():
    """Pid file of this process's memory holder: JEV_HOLD_PIDFILE when set (2026-10-08: several processes share one GPU,
    each with its own slot -- it releases and later re-creates only its slot's holder), else
    runs/.gpu_hold_<first visible GPU>.pid."""
    if os.environ.get("JEV_HOLD_PIDFILE"):
        return Path(os.environ["JEV_HOLD_PIDFILE"]).resolve()
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    return Path(__file__).resolve().parents[2] / "runs" / f".gpu_hold_{gpu}.pid"


def _release_holder(timeout=120):
    """Terminate the memory holder of this GPU (scripts/gpu_hold.py) and wait until it has exited."""
    import signal
    pf = _hold_pidfile()
    try:
        pid = int(pf.read_text().strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(json.dumps({"gpu_holder_released": pid, "seconds": round(time.time() - t0, 2)}), flush=True)
            return
        time.sleep(0.05)
    print(json.dumps({"gpu_holder_still_running": pid}), flush=True)


def _spawn_holder(gb):
    """At exit (also after an error): start scripts/gpu_hold.py and wait until it is ready; it takes the memory as soon
    as this process ends."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    (root / "logs").mkdir(exist_ok=True)
    with open(root / "logs" / "gpu_hold.log", "a") as log:
        p = subprocess.Popen([sys.executable, str(root / "scripts" / "gpu_hold.py"), "--gb", str(gb),
                              "--wait-pid", str(os.getpid()), "--pidfile", str(_hold_pidfile())],
                             stdout=log, stderr=log, start_new_session=True, cwd=str(root))
    t0 = time.time()                                    # exit only once the holder has its CUDA context (pid file)
    while time.time() - t0 < 120 and p.poll() is None:
        try:
            if int(_hold_pidfile().read_text().strip()) == p.pid:
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.1)


def reserve_gpu_memory():
    """JEV_RESERVE_GB=<n> (shared machines; unset = no-op): allocate n GB on cuda:0 once, before any model is built, and
    free it into PyTorch's caching allocator. Later allocations of this process are carved out of that block instead
    of asking the driver, so another user's job started later cannot take the memory this run needs.
    JEV_HOLD=1 additionally keeps the memory between processes and after failures: at exit this process starts
    scripts/gpu_hold.py (JEV_HOLD_GB, default JEV_RESERVE_GB), which takes the memory when the process has ended; the
    next process terminates the holder right before its own reservation (CUDA already initialised, so the gap is short)."""
    gb = float(os.environ.get("JEV_RESERVE_GB", 0) or 0)
    if gb <= 0:
        return
    import torch
    if not torch.cuda.is_available():
        return
    torch.zeros(1, device="cuda:0")                  # CUDA context first, then take over from the holder
    _release_holder()
    want, got = gb, None
    t_wait = time.time() + float(os.environ.get("JEV_RESERVE_WAIT_S", 30))
    while gb >= want / 2:                                # not enough free memory: take what is there, 5 GB less each try
        try:
            x = torch.empty(int(gb * 2 ** 30), dtype=torch.uint8, device="cuda:0")
            del x
            got = gb
            break
        except torch.OutOfMemoryError:
            if time.time() < t_wait:                     # a holder just told to exit (by this process or a sibling
                time.sleep(0.25)                         # started at the same time) may still be freeing its memory
                continue
            gb -= 5
    torch.cuda.reset_peak_memory_stats(0)               # gpu_mem_peak_gb then reports real use, not the reservation
    print(json.dumps({"reserved_gb": got, "requested_gb": want,
                      "torch_reserved_gb": round(torch.cuda.memory_reserved(0) / 2 ** 30, 1)}), flush=True)
    if os.environ.get("JEV_HOLD") == "1":
        import atexit
        atexit.register(_spawn_holder, float(os.environ.get("JEV_HOLD_GB") or want))


def gpu_mem_stats():
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    except ImportError:
        pass
    return {}
