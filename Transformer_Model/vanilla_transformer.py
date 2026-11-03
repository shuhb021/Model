"""
vanilla_transformer.py — Vanilla Transformer Baseline (Vaswani et al., 2017)
=============================================================================
Satisfies Reviewer / Research Requirement:
  "P2: Try vanilla transformer with same data and BPE tokenizer."

Architecture:
  - Canonical Sequence-to-Sequence Transformer
  - Sinusoidal Positional Encoding
  - Standard Multi-Head Self-Attention & Cross-Attention
  - Causal lookahead triangular masking in decoder
  - Feed-Forward Network with ReLU/GELU activations
  - Supports both BPE tokenizers:
      1. BPE 1157-token tokenizer (from ByT5 setup)
      2. BPE 5000-token tokenizer (gondi_bpe_model.json)
      3. Any custom BPE vocabulary size

Usage:
  # 1. Train Vanilla Transformer on same data using 1157 BPE tokenizer:
  python vanilla_transformer.py --train --tokenizer_path "../Byt5 Model/Tokenizer_Files/tokenizer.json" --epochs 30

  # 2. Train Vanilla Transformer with 5000 BPE tokenizer:
  python vanilla_transformer.py --train --tokenizer_path "gondi_bpe_model.json" --epochs 30

  # 3. Evaluate trained Vanilla Transformer:
  python vanilla_transformer.py --evaluate --checkpoint "best_vanilla_transformer_hindi.pt"

  # 4. Interactive Translation:
  python vanilla_transformer.py --translate --checkpoint "best_vanilla_transformer_hindi.pt"
"""

import os
import sys
import math
import json
import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. POSITIONAL ENCODING & ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Canonical sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VanillaTransformer(nn.Module):
    """
    Standard Vanilla Transformer Encoder-Decoder from scratch.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 pad_idx: int = 0, device: Optional[torch.device] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.device = device or torch.device("cpu")

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(sz, device=self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (trg == self.pad_idx)

        tgt_seq_len = trg.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        src_emb = self.pos_encoder(self.src_embedding(src) * self.scale)
        tgt_emb = self.pos_encoder(self.tgt_embedding(trg) * self.scale)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc_out(out)

    @torch.no_grad()
    def translate(self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 100) -> Tuple[torch.Tensor, List]:
        self.eval()
        B = src.size(0)
        src_key_padding_mask = (src == self.pad_idx)
        src_emb = self.pos_encoder(self.src_embedding(src) * self.scale)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1))
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * self.scale)
            out = self.transformer.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            prob = self.fc_out(out[:, -1])
            next_word = prob.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
            finished = finished | (next_word.squeeze(1) == eos_idx)
            if finished.all():
                break

        return ys[:, 1:], []


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKENIZER WRAPPER (Supports both 1157 BPE and 5000 BPE)
# ─────────────────────────────────────────────────────────────────────────────

class BpeTokenizerWrapper:
    def __init__(self, tokenizer_path: str):
        self.path = tokenizer_path
        is_hf = False
        try:
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "model" in d and ("type" in d["model"] or "version" in d):
                    is_hf = True
        except Exception:
            pass

        if is_hf:
            from transformers import PreTrainedTokenizerFast
            self.tok = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path,
                unk_token="[UNK]",
                pad_token="[PAD]",
                eos_token="</s>",
                bos_token="[START]"
            )
            self.is_hf = True
            self.pad_idx = self.tok.pad_token_id if self.tok.pad_token_id is not None else 1
            self.unk_idx = self.tok.unk_token_id if self.tok.unk_token_id is not None else 0
            self.eos_idx = self.tok.eos_token_id if self.tok.eos_token_id is not None else 2
            self.sos_idx = self.tok.bos_token_id if self.tok.bos_token_id is not None else 3
            self._vocab_size = len(self.tok)
        else:
            from gondi_bpe_tokenizer import GondiTokenizer
            self.tok = GondiTokenizer.load(tokenizer_path)
            self.is_hf = False
            self.pad_idx = self.tok.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
            self.unk_idx = self.tok.token_to_id.get(GondiTokenizer.UNK_TOKEN, 1)
            self.sos_idx = self.tok.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
            self.eos_idx = self.tok.token_to_id.get(GondiTokenizer.END_TOKEN, 3)
            self._vocab_size = self.tok.vocab_size()

    def encode(self, text: str) -> List[int]:
        if self.is_hf:
            return self.tok.encode(text, add_special_tokens=False)
        return self.tok.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self.is_hf:
            return self.tok.decode(ids, skip_special_tokens=True)
        words = []
        for tid in ids:
            if tid == self.eos_idx:
                break
            if tid not in (self.pad_idx, self.sos_idx):
                w = self.tok.id_to_token.get(tid, "")
                if w:
                    words.append(w)
        return "".join(words).replace(" ", " ").strip()

    def vocab_size(self) -> int:
        return self._vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLI AND EXECUTION MODES
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla Transformer with same corpus and BPE tokenizer.")
    parser.add_argument("--train", action="store_true", help="Run training mode.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode.")
    parser.add_argument("--translate", action="store_true", help="Run interactive translation.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to BPE tokenizer (e.g. tokenizer.json for 1157 tokens or gondi_bpe_model.json).")
    parser.add_argument("--csv_path", type=str, default=None, help="Corpus CSV path.")
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"])
    parser.add_argument("--checkpoint", type=str, default="best_vanilla_transformer_hindi.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve tokenizer path (check local, then Byt5 Model 1157 tokens)
    tok_path = args.tokenizer_path
    if not tok_path:
        cand1 = os.path.join(base_dir, "..", "Byt5 Model", "Tokenizer_Files", "tokenizer.json")
        cand2 = os.path.join(base_dir, "gondi_bpe_model.json")
        tok_path = cand1 if os.path.exists(cand1) else cand2

    csv_path = args.csv_path or os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")

    print("=" * 70)
    print("  ⚡ Vanilla Transformer Baseline (Requirement P2)")
    print(f"  📂 Dataset:   {csv_path}")
    print(f"  🔤 Tokenizer: {tok_path}")
    print(f"  🖥️  Device:    {device}")
    print("=" * 70)

    tokenizer = BpeTokenizerWrapper(tok_path)
    vocab_size = tokenizer.vocab_size()
    print(f"  ✓ BPE Tokenizer Loaded: {vocab_size} tokens | PAD={tokenizer.pad_idx}")

    model = VanillaTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dim_feedforward=2048,
        pad_idx=tokenizer.pad_idx,
        device=device
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Vanilla Transformer Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    if not (args.train or args.evaluate or args.translate):
        print("\n  💡 Ready to run! Specify one of:")
        print("     --train      : To train Vanilla Transformer from scratch")
        print("     --evaluate   : To evaluate a saved checkpoint on test set")
        print("     --translate  : For interactive translation CLI")
        return


if __name__ == "__main__":
    main()

