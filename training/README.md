# LoopSeek: System I training

Training code for System I of **LoopSeek**, from the paper
*LoopSeek: Decoupling System One Search from System Two Answering for Long Video Understanding*
([arXiv:2611.28635](https://arxiv.org/abs/2611.28635)).

System I searches a long video for the frames that a frozen multimodal LLM (System II) needs to answer a
multiple-choice question. Its roles are LoRA adapters on [OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B):

| Role in the paper | Name in the code | Decides |
|---|---|---|
| Controller | `controller` | the next action on the window tree: `Refine`, `Switch`, `Up` or `Inspect` |
| Locator | `localize` (grounding adapter) | which child window of the active window to enter |
| Selector | `frame` (grounding adapter) | which frames of the active window to inspect |
| Verifier | `verifier` | whether an inspected frame shows evidence, and where the evidence lies relative to the active window |

The Locator and the Selector share one adapter. System II,
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), is frozen: during training it filters the
training questions and answers at the end of every collected episode.

This folder holds the training code; the inference and evaluation code is in [`../inference`](../inference).

## Requirements

- Linux with NVIDIA GPUs. The settings of the paper's run are for 8 GPUs with 80 GB of memory or more each.
- conda. The environment is Python 3.12 with PyTorch 2.14 (CUDA 12.6); all versions are pinned in
  `configs/env_jev_fa.lock.txt`. System II runs in a separate vLLM environment (`configs/env_vllm.requirements.txt`).
- Access to the Hugging Face models above; their revisions are pinned in `scripts/download_models.py`.

## Setup

```bash
cp run.env.example run.env     # fill in the paths
bash run.sh setup              # conda env, System II's vLLM env, model download, CPU tests
```

## Data

Training questions come from three sources (paths in `run.env`):

- **CG-Bench** (Hugging Face dataset `CG-Bench/CG-Bench`, `scripts/prepare_cgbench.py`). Questions outside CG-Bench
  mini are the training pool; 700 of them are held out as dev. CG-Bench mini is the test split.
- **LV-Haystack** (Ego4D videos, `scripts/prepare_lvhaystack.py`).
- **VideoITG-40K** (short videos, `scripts/fetch_videoitg.py`). Every one of its 84,139 multiple-choice training
  questions goes to the filter; dev is 300 questions drawn by duration (30-60 s 20%, 1-2 min 30%, 2-3 min 50%) from
  those of 300 held-out videos. They are trained as a second profile with the frame budget halved (`configs/short.yaml`).

The data step builds the manifests under `data/manifests/` and filters the training questions with the four-stage
filter of VTS (`scripts/filter_cgbench_vts.py`), with System II as the answerer: a question is kept when its annotated
clue covers at most 30% of the video, System II cannot answer it from the text alone or from frames outside the clue,
and can answer it from frames inside the clue. Dev and test questions are only tagged with the filter's verdict.

## Training

```bash
bash run.sh smoke              # optional: every step on 16 / 8 / 8 questions, to check that the pipeline runs
bash run.sh sft                # data step, then the steps below; run the same command again to resume
bash run.sh status             # finished steps of $TAG
```

`run.sh sft` runs the steps of `scripts/train_pipeline.sh`:

| Step | What it does |
|---|---|
| `data` | manifests of the three sources; question filter on train (skipped with `PREPARED_DATA=1`) |
| `s1_collect` | expert trajectories on train and dev: an expert controller that follows the annotated evidence, grounding from the evidence or from zero-shot OneJev-4B, and verifier answers taken from the evidence |
| `s1_data`, `s1_train`, `s2_train` | SFT of the grounding adapter (`grounding1`) and the verifier (`verifier1`) |
| `s3_collect` | expert controller with the round-1 adapters, wrong windows injected at random |
| `s4_train` | SFT of `grounding2` and `verifier2` on these states, with replay of round 1 |
| `s5_collect`, `s5_train` | expert controller with the round-2 adapters; SFT of the controller on the expert's actions |
| `s5_dagger 1` | DAgger: the trained controller acts (the expert with probability 0.3), the expert labels every state, and the controller is trained further on all rows so far |

Collections run one shard per GPU of `GPUS`; SFT runs on all of them with `SFT_GPUS=all`. Every step leaves a marker
in `runs/.done/`, and a rerun skips finished steps. The result is the policy set `runs/<TAG>_ps3_r1.json`, which lists
the adapter checkpoints of the four roles.

`run.env.example` holds the paper's settings (one DAgger round, expert mixing probability 0.3, wrong-window
injection 0.3), and `configs/train_8gpu.yaml` the batch sizes of its 8-GPU run (32 decisions per SFT step,
25 warm-up steps). `configs/default.yaml` holds every other setting.

## Tests

```bash
python -m pytest tests -q      # CPU only
```

## Citation

```bibtex
@article{wang2026loopseek,
  title   = {LoopSeek: Decoupling System One Search from System Two Answering for Long Video Understanding},
  author  = {Wang, Chengzhong and Lin, Lang and Liu, Xinyang and Yin, Yueqin and Zhou, Mingyuan and An, Shengwei},
  journal = {arXiv preprint arXiv:2611.28635},
  year    = {2026}
}
```
