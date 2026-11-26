"""Minimal LoRA for the NanoJev backbone (update.md §2, §6: rank 16, alpha 32, dropout 0).

y = W x + (alpha / r) * B A x, with W frozen, A ~ kaiming-uniform, B = 0 at init, so a freshly
wrapped model computes exactly the original function.
"""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        a = self.lora_A.to(x.dtype) if not torch.is_autocast_enabled() else self.lora_A
        b = self.lora_B.to(x.dtype) if not torch.is_autocast_enabled() else self.lora_B
        return out + nn.functional.linear(nn.functional.linear(self.drop(x), a), b) * self.scaling


def apply_lora(model: nn.Module, targets, r, alpha, dropout=0.0):
    """Wrap every nn.Linear whose attribute name is in `targets`; freeze everything else in `model`.
    Returns the sorted list of wrapped module names (recorded in the run config)."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return sorted(wrapped)


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]
