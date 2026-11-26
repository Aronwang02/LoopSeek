"""Inference speedups that fuse memory-bound elementwise work of the OneJev-4B (Qwen3.5) backbone.

policy.compile_norms (2026-11-05, speedup F1): Qwen3_5RMSNorm and Qwen3_5RMSNormGated run eagerly as 6-8 small kernels each
(fp32 cast, pow, mean, rsqrt, mul, weight mul, cast back); a batch-1 question runs ~3.9k norm calls (~104 per forward x ~37
forwards), about a quarter of its ~95k kernels (profile 11-05). torch.compile(dynamic=True) of the two unchanged forward
functions fuses each call into one Triton kernel. Same math in fp32; the reduction order may differ, so results can differ
in the last bits (checked by argmax agreement, not bit equality)."""
import torch


def compile_qwen35_norms():
    """Replace the forward of both Qwen3.5 norm classes by compiled versions (idempotent; affects every instance)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
    import torch._inductor.config as inductor_config
    # the gated norm rounds to bf16 in the middle (weight * x.to(bf16)); inductor drops such intermediate casts unless asked
    # to emulate them (CPU check 11-05: 65% of elements bit-equal without, 100% with)
    inductor_config.emulate_precision_casts = True
    done = []
    for cls in (Q.Qwen3_5RMSNorm, Q.Qwen3_5RMSNormGated):
        if getattr(cls, "_jev_compiled_forward", False):
            continue
        cls.forward = torch.compile(cls.forward, dynamic=True)
        cls._jev_compiled_forward = True
        done.append(cls.__name__)
    return done
