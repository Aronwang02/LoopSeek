"""Hold GPU memory between runs on a shared machine (JEV_HOLD=1, src/common/runtime.reserve_gpu_memory). Started by a
GPU process when it exits (also after an error): initialises CUDA, waits for that process to end, then takes --gb GB on
cuda:0 in 1 GB pieces (retrying while memory is still being freed) and keeps it until SIGTERM. The next GPU process
terminates the holder (pid file) right before reserving the memory itself. One holder per pid file: a holder whose pid
file names another process (two holders started at the same moment; the last writer wins) exits.
usage (normally automatic): python scripts/gpu_hold.py --gb 100 --wait-pid <pid> --pidfile runs/.gpu_hold_<gpu>.pid"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, required=True)
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--pidfile", required=True)
    args = ap.parse_args()
    pf = Path(args.pidfile)
    log = lambda **kw: print(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(), **kw}), flush=True)  # noqa: E731
    try:
        other = int(pf.read_text().strip())
    except (OSError, ValueError):
        other = None
    if other and other != os.getpid() and alive(other):
        log(event="exit", reason=f"holder {other} is already running")
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    import torch
    torch.zeros(1, device="cuda:0")                       # CUDA context before the previous process frees its memory
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(str(os.getpid()))                       # = ready: the exiting process waits for this before it exits

    def mine():
        try:
            return pf.read_text().strip() == str(os.getpid())
        except OSError:
            return False

    while args.wait_pid and alive(args.wait_pid):
        time.sleep(0.01)
    if not mine():
        log(event="exit", reason="pid file names another holder")
        return
    t_free = time.time()
    chunks, target, last = [], int(args.gb), 0.0
    while len(chunks) < target:
        try:
            chunks.append(torch.empty(2 ** 30, dtype=torch.uint8, device="cuda:0"))
        except torch.OutOfMemoryError:
            if not mine():
                log(event="exit", reason="pid file names another holder", held_gb=len(chunks))
                return
            if time.time() - last > 300:
                log(event="waiting", held_gb=len(chunks), target_gb=target)
                last = time.time()
            time.sleep(0.05 if time.time() - t_free < 30 else 2)   # memory of the exiting process is freed shortly
    log(event="holding", held_gb=len(chunks), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        seconds_after_previous_exit=round(time.time() - t_free, 2))
    try:
        while mine():
            time.sleep(1)
        log(event="exit", reason="pid file names another holder", held_gb=len(chunks))
    finally:
        if pf.exists() and pf.read_text().strip() == str(os.getpid()):
            pf.unlink()


if __name__ == "__main__":
    main()
