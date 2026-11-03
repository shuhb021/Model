"""
LoRA (Low-Rank Adaptation) for Parameter-Efficient Fine-Tuning
──────────────────────────────────────────────────────────────
Freezes base parameters θ₀ and injects low-rank adapters:
    W = W₀ + ΔW = W₀ + (α/r)(B × A)

Configuration:
    rank  (r)  : 16
    alpha (α)  : 32
    Target     : q_proj, k_proj, v_proj, out_proj (self + cross attention)
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adapter.
    The base weight W₀ is frozen; only A and B are trained.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.alpha        = alpha
        self.scaling      = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B initialised to zero → ΔW = 0 at start of training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA into base weight for zero-overhead inference."""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def trainable_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


import torch.nn.functional as F


def apply_lora_to_model(model: nn.Module,
                         target_modules: Optional[List[str]] = None,
                         rank: int = 16,
                         alpha: float = 32.0,
                         dropout: float = 0.05) -> nn.Module:
    """
    Walk the model and replace target Linear layers with LoRALinear.
    Base parameters are frozen; only LoRA adapters are trainable.

    Args:
        model          : The AsymmetricNMT model
        target_modules : Module name substrings to replace.
                         Defaults to all four attention projections.
        rank           : LoRA rank r
        alpha          : LoRA scaling alpha
        dropout        : Dropout on LoRA input

    Returns:
        model with LoRA injected and base params frozen
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    # Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, attr = name.rsplit(".", 1)
                parent = _get_submodule(model, parent_name)

                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None,
                )
                # Copy pretrained weights into frozen base
                with torch.no_grad():
                    lora_layer.weight.copy_(module.weight)
                    if module.bias is not None:
                        lora_layer.bias.data.copy_(module.bias.data)

                setattr(parent, attr, lora_layer)
                replaced += 1
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[OK] LoRA applied to {replaced} modules | "
          f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def _get_submodule(model: nn.Module, target: str) -> nn.Module:
    parts = target.split(".")
    sub = model
    for part in parts:
        sub = getattr(sub, part)
    return sub


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from model.architecture import AsymmetricNMT

    model = AsymmetricNMT()
    print(f"Before LoRA - trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model = apply_lora_to_model(model, rank=16, alpha=32)

    src = torch.randint(1, 32000, (2, 20))
    tgt = torch.randint(1, 32000, (2, 15))
    logits = model(src, tgt)
    print(f"[OK] LoRA forward pass OK | logits: {tuple(logits.shape)}")
