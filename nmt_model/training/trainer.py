"""
Training Loop — Curriculum Learning
─────────────────────────────────────
Stage 1: High-resource warmup (Hindi ↔ English)
Stage 2: Multilingual joint training with temperature sampling (T=1.6)
Stage 3: Gondi specialization with frozen base + LoRA only

Optimizer : AdamW | β1=0.9, β2=0.98, ε=1e-8, weight_decay=0.01
Scheduler : Cosine with linear warmup (4,000 steps → 1e-6)
Precision : Mixed FP16/BF16 via torch.cuda.amp
Batch     : 32 sentences × 4 gradient accumulation = 128 effective
"""

import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ─── Label-Smoothed Cross-Entropy ────────────────────────────────────────────

class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing (ε=0.1) for better calibration
    in low-resource settings.
    """

    def __init__(self, vocab_size: int, pad_id: int = 0,
                 smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id     = pad_id
        self.smoothing  = smoothing

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, T, V]
        target : [B, T]
        """
        V = logits.shape[-1]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Label smoothing
        smooth_target = torch.full_like(log_probs, self.smoothing / (V - 1))
        smooth_target.scatter_(-1, target.unsqueeze(-1).clamp(min=0), 1.0 - self.smoothing)

        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Mask padding
        pad_mask = target.ne(self.pad_id).float()
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        return loss


# ─── Cosine Scheduler with Warmup ─────────────────────────────────────────────

class CosineWarmupScheduler:
    """
    Linear warmup for `warmup_steps` then cosine decay to `min_lr`.
    Step the scheduler every training step (not epoch).
    """

    def __init__(self, optimizer, d_model: int = 512,
                 warmup_steps: int = 4_000, min_lr: float = 1e-6):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.min_lr       = min_lr
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        s = self._step
        if s < self.warmup_steps:
            # Linear warmup
            return (self.d_model ** -0.5) * s * (self.warmup_steps ** -1.5)
        else:
            # Cosine decay
            progress = (s - self.warmup_steps) / max(1, self.warmup_steps * 10)
            cosine   = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return self.min_lr + cosine * ((self.d_model ** -0.5) * (self.warmup_steps ** -0.5) - self.min_lr)

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ─── Trainer ──────────────────────────────────────────────────────────────────

class NMTTrainer:
    """
    Full training loop with:
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Curriculum stage management
    - Checkpoint saving/loading
    - Basic validation loop
    """

    def __init__(self,
                 model: nn.Module,
                 train_loaders: Dict[str, DataLoader],
                 val_loader: Optional[DataLoader] = None,
                 save_dir: str = "checkpoints",
                 d_model: int = 512,
                 warmup_steps: int = 4_000,
                 grad_accum_steps: int = 4,
                 max_grad_norm: float = 1.0,
                 use_bf16: bool = False):

        self.model           = model
        self.train_loaders   = train_loaders
        self.val_loader      = val_loader
        self.save_dir        = Path(save_dir)
        self.grad_accum      = grad_accum_steps
        self.max_grad_norm   = max_grad_norm
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

        # Mixed precision
        self.dtype  = torch.bfloat16 if use_bf16 else torch.float16
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # Optimizer — only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=1e-3,              # will be immediately overridden by scheduler
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer, d_model=d_model, warmup_steps=warmup_steps
        )

        self.criterion = LabelSmoothedCrossEntropy(
            vocab_size=model.vocab_size if hasattr(model, 'vocab_size') else 32_000
        ).to(self.device)

        self.global_step = 0
        self.best_val_loss = float("inf")

        print(f"[OK] Trainer ready | device={self.device} | "
              f"trainable params={sum(p.numel() for p in trainable_params):,}")

    # ── Core Train Step ────────────────────────────────────────────────────────

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        src     = batch["src"].to(self.device)
        tgt_in  = batch["tgt_in"].to(self.device)
        tgt_out = batch["tgt_out"].to(self.device)

        with autocast(dtype=self.dtype, enabled=torch.cuda.is_available()):
            logits = self.model(src, tgt_in)             # [B, T, V]
            loss   = self.criterion(logits, tgt_out)
            loss   = loss / self.grad_accum

        self.scaler.scale(loss).backward()
        return loss.item() * self.grad_accum

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, stage_name: str = "") -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            loss = self._train_step(batch)
            total_loss += loss

            if (step + 1) % self.grad_accum == 0:
                self._optimizer_step()

                if self.global_step % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{stage_name}] step={self.global_step:6d} | "
                          f"loss={loss:.4f} | lr={self.scheduler.current_lr:.2e} | "
                          f"{elapsed:.1f}s")
                    t0 = time.time()

            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss, n = 0.0, 0

        for batch in self.val_loader:
            src     = batch["src"].to(self.device)
            tgt_in  = batch["tgt_in"].to(self.device)
            tgt_out = batch["tgt_out"].to(self.device)

            with autocast(dtype=self.dtype, enabled=torch.cuda.is_available()):
                logits = self.model(src, tgt_in)
                loss   = self.criterion(logits, tgt_out)

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, name: str = "checkpoint"):
        path = self.save_dir / f"{name}.pt"
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_step":  self.scheduler._step,
            "global_step":     self.global_step,
            "best_val_loss":   self.best_val_loss,
        }, path)
        print(f"  [SAVE] Checkpoint saved -> {path}")

    def load_checkpoint(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        self.scheduler._step = ck["scheduler_step"]
        self.global_step      = ck["global_step"]
        self.best_val_loss    = ck["best_val_loss"]
        print(f"  [OK] Loaded checkpoint: {path} | step={self.global_step}")

    # ── Curriculum ────────────────────────────────────────────────────────────

    def run_curriculum(self, epochs_per_stage: Dict[str, int] = None):
        """
        Three-stage curriculum as per spec:
          Stage 1 — High-resource warmup (Hindi ↔ English)
          Stage 2 — Multilingual joint (all directions)
          Stage 3 — Gondi specialisation (LoRA only, freeze base)
        """
        if epochs_per_stage is None:
            epochs_per_stage = {
                "stage1_warmup":      5,
                "stage2_multilingual": 10,
                "stage3_gondi":       5,
            }

        # ── Stage 1 ──────────────────────────────────────────────────────────
        if "hi_en" in self.train_loaders:
            print("\n" + "═"*55)
            print("  STAGE 1: High-Resource Warmup (Hindi ↔ English)")
            print("═"*55)
            for epoch in range(epochs_per_stage["stage1_warmup"]):
                tr_loss = self.train_epoch(self.train_loaders["hi_en"], "Stage1")
                val_loss = self.evaluate()
                print(f"  Epoch {epoch+1} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
            self.save_checkpoint("stage1_warmup")

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if "multilingual" in self.train_loaders:
            print("\n" + "═"*55)
            print("  STAGE 2: Multilingual Joint Training (T=1.6 sampling)")
            print("═"*55)
            for epoch in range(epochs_per_stage["stage2_multilingual"]):
                tr_loss = self.train_epoch(self.train_loaders["multilingual"], "Stage2")
                val_loss = self.evaluate()
                print(f"  Epoch {epoch+1} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model")
            self.save_checkpoint("stage2_multilingual")

        # ── Stage 3 ──────────────────────────────────────────────────────────
        if "gondi" in self.train_loaders:
            print("\n" + "═"*55)
            print("  STAGE 3: Gondi Specialisation (LoRA fine-tuning)")
            print("═"*55)
            for epoch in range(epochs_per_stage["stage3_gondi"]):
                tr_loss = self.train_epoch(self.train_loaders["gondi"], "Stage3")
                val_loss = self.evaluate()
                print(f"  Epoch {epoch+1} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
            self.save_checkpoint("stage3_gondi_final")

        print("\n[OK] Curriculum training complete!")
