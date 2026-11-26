# LoopSeek: inference

Runs LoopSeek on long-video multiple-choice questions: System I (the released adapters on OneJev-4B) searches each video
for frames, and System II (Qwen3-VL-8B-Instruct) answers once from 32 frames. With the steps below it evaluates the five
benchmarks of the paper: Video-MME (no subtitles), MLVU dev, LongVideoBench val, LVBench and CG-Bench mini.

## Requirements

- Linux with NVIDIA GPUs. Each inference process runs System I and System II, either on one GPU with 80 GB or more. Processes run in parallel and share one question queue.
- conda. The environment is Python 3.12 with PyTorch 2.14 (CUDA 12.6).
## Setup

```bash
cp run.env.example run.env     # fill in the paths and the GPUs
bash run.sh setup              # conda env, System II's vLLM env, OneJev-4B / Qwen3-VL-8B download, CPU tests
bash run.sh weights            # the adapters from huggingface.co/Chengzhong-Wang/LoopSeek
bash run.sh data               # the five benchmarks, downloaded from their official sources -> $DATA/manifests/bench5
```

## Run

```bash
bash run.sh smoke             
bash run.sh bench5             
bash run.sh infer <manifest dir> <file> <name>  
```
### Another System II

The search does not depend on System II, which is called once at the end, so the frames of one search run can be
answered by another model:

```bash
bash run.sh answer Qwen/Qwen2.5-VL-7B-Instruct     # Qwen3-VL, Qwen2.5-VL or Qwen2-VL checkpoints
```

For other model families, `scripts/export_answer_frames.py` writes the 32 frame times of every question,
your code answers, and `scripts/import_answers.py` scores the replies the same way.

## Settings

`configs/default.yaml` holds every setting and `configs/loopseek.yaml` the inference settings of the paper over it:
an overview of K_o = 96 frames, of which the Verifier gives the K_a = 24 most likely to show the evidence to System II,
and a search of at most B = 24 actions that keeps at most K_d = 8 frames, picking m = 4 of M = 32 candidates per
`Inspect`. The speedups A1-A4, V1 and V2 in `configs/loopseek.yaml` keep the decisions unchanged up to near ties.
One more speedup, E1, is off; see below.

### Speedup E1: one folded adapter per role (off by default)

System I is one frozen OneJev-4B with three LoRA adapters: one for the Controller, one shared by the Locator and the
Selector, and one for the Verifier. They adapt 248 weight matrices of OneJev-4B.
With the default settings, the three adapters share one bf16 copy of these matrices with an adapter folded in. Whenever the next decision belongs to another role, all 248 matrices are
folded again; with one question in flight this happens about 16 times per MLVU question. E1
(`policy.merge_lora_per_role: true`) keeps one folded copy per adapter, so a change of role costs nothing.

- Results do not change. With E1 the forwards are identical bit for bit to the default ones.
- Memory: three folded copies instead of one, about 14 GB more on the GPU that runs System I. That is why E1 is off
  by default: with `configs/two_gpu.yaml` on 40 GB GPUs, System I alone peaks at about 30 GB with 8 questions in
  flight, and on one GPU it shares the memory with System II.
- To turn it on, set `merge_lora_per_role: true` under `policy:` in `configs/loopseek.yaml` (E1 works on top of A1,
  `merge_lora_inference: true`, which is already set there), on GPUs with that much free memory.

## Tests

```bash
python -m pytest tests -q      # CPU only
```
