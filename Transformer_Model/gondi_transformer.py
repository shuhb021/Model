"""
gondi_transformer.py — Standard Transformer Seq2Seq Baselines for Gondi Translation
===================================================================================
Standard PyTorch Transformer trained completely from scratch:
  - Positional Encoding: Sinusoidal Positional Encoding (Vaswani et al.)
  - Multi-Head Self-Attention and Cross-Attention with causal lookahead masking
  - Transformer-Small (d=256, 4 layers, 4 heads, d_ff=1024)
  - Transformer-Base  (d=512, 6 layers, 8 heads, d_ff=2048)
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
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


class TransformerSeq2Seq(nn.Module):
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
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(sz, device=self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 1.0,
                src_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        logits = self.fc_out(out)
        return logits

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embedding(src) * self.scale)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * self.scale)
        return self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    @torch.no_grad()
    def translate(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                  max_len: int = 120, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.eval()
        B = src.size(0)

        src_key_padding_mask = (src == self.pad_idx)
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            tgt_seq_len = ys.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

            out = self.decode(
                tgt=ys,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask
            )

            prob = self.fc_out(out[:, -1])
            next_word = prob.argmax(dim=-1, keepdim=True)

            ys = torch.cat([ys, next_word], dim=1)
            finished = finished | (next_word.squeeze(1) == eos_idx)
            if finished.all():
                break

        return ys[:, 1:], []


def build_model(gondi_vocab_size: int, target_vocab_size: int, device: torch.device,
                model_size: str = "small", pad_idx: int = 0) -> TransformerSeq2Seq:
    if model_size == "small":
        return TransformerSeq2Seq(
            vocab_size=gondi_vocab_size,
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            pad_idx=pad_idx,
            device=device
        ).to(device)
    else:  # base
        return TransformerSeq2Seq(
            vocab_size=gondi_vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            pad_idx=pad_idx,
            device=device
        ).to(device)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_small = build_model(5000, 5000, dev, "small")
    m_base = build_model(5000, 5000, dev, "base")
    print(f"Transformer-Small params: {sum(p.numel() for p in m_small.parameters() if p.requires_grad):,}")
    print(f"Transformer-Base  params: {sum(p.numel() for p in m_base.parameters() if p.requires_grad):,}")

