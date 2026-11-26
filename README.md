# LoopSeek

Code for the paper
*LoopSeek: Decoupling System One Search from System Two Answering for Long Video Understanding*
([arXiv:2611.28635](https://arxiv.org/abs/2611.28635)).

LoopSeek answers multiple-choice questions about long videos in two parts. System I, three LoRA adapters on
[OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B), searches the video on a tree of windows for the frames that hold
the evidence; System II, a frozen multimodal LLM ([Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
in the paper), answers once from those frames. The adapters are at
[huggingface.co/Chengzhong-Wang/LoopSeek](https://huggingface.co/Chengzhong-Wang/LoopSeek).

| Folder | Contents |
|---|---|
| [`inference/`](inference) | LoopSeek with the released adapters on multiple-choice video questions; the evaluation on Video-MME, MLVU, LongVideoBench, LVBench and CG-Bench mini; answers of other System II models from the same frames |
| [`training/`](training) | Training of System I: question filtering, staged SFT of the roles with an evidence expert, DAgger |

Each folder has its own setup, configuration, README and tests. The datasets are not part of this repository: the
scripts download them from their official sources.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).

## Citation

```bibtex
@article{wang2026loopseek,
  title   = {LoopSeek: Decoupling System One Search from System Two Answering for Long Video Understanding},
  author  = {Wang, Chengzhong and Lin, Lang and Liu, Xinyang and Yin, Yueqin and Zhou, Mingyuan and An, Shengwei},
  journal = {arXiv preprint arXiv:2611.28635},
  year    = {2026}
}
```
