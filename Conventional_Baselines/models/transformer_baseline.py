"""
Standard Transformer Seq2Seq Baselines (Small and Base) — Gondi Language Translation
=====================================================================================
Implements standard Transformer (Vaswani et al.) trained completely from scratch.

Presets:
  - Transformer-Small: d_model=256, nhead=4, 4 enc layers, 4 dec layers, d_ff=1024, dropout=0.1
  - Transformer-Base:  d_model=512, nhead=8, 6 enc layers, 6 dec layers, d_ff=2048, dropout=0.1

Features:
  - Sinusoidal Positional Encoding
  - Standard Multi-Head Self-Attention and Cross-Attention
  - Causal lookahead mask and padding masks
  - Autoregressive greedy/beam translation
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding (Vaswani et al., 2017).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    Standard Transformer Seq2Seq trained from scratch.
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

        # Source and Target embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # PyTorch standard Transformer
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

        # Output projection to target vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

        # Initialize weights with Xavier uniform
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates causal mask."""
        return nn.Transformer.generate_square_subsequent_mask(sz, device=self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 1.0,
                src_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src: [batch, src_len]
        trg: [batch, trg_len]  (includes <SOS> at idx 0 and <EOS> at end)
        During training, trg input to decoder is trg[:, :-1], predicting trg[:, 1:]
        """
        # Padding masks: True for positions with PAD
        src_key_padding_mask = (src == self.pad_idx)  # [B, S]
        tgt_key_padding_mask = (trg == self.pad_idx)  # [B, T]

        tgt_seq_len = trg.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        # Embeddings + Positional Encodings
        src_emb = self.pos_encoder(self.src_embedding(src) * self.scale)
        tgt_emb = self.pos_encoder(self.tgt_embedding(trg) * self.scale)

        # Transformer forward pass
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )  # [B, T, d_model]

        logits = self.fc_out(out)  # [B, T, vocab_size]
        return logits

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes source sequence to memory representation."""
        src_emb = self.pos_encoder(self.src_embedding(src) * self.scale)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decodes target sequence given encoder memory."""
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
        """
        Autoregressive greedy generation.
        """
        self.eval()
        B = src.size(0)

        src_key_padding_mask = (src == self.pad_idx)
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        # Start decoder input with <SOS>
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

            # Project last step to vocabulary
            prob = self.fc_out(out[:, -1])  # [B, vocab_size]
            next_word = prob.argmax(dim=-1, keepdim=True)  # [B, 1]

            ys = torch.cat([ys, next_word], dim=1)

            finished = finished | (next_word.squeeze(1) == eos_idx)
            if finished.all():
                break

        # Output tokens excluding initial <SOS>
        generated = ys[:, 1:]
        return generated, []


def build_transformer_small(vocab_size: int, device: torch.device,
                            pad_idx: int = 0) -> TransformerSeq2Seq:
    """
    Standard Transformer-Small preset:
      - d_model = 256
      - nhead = 4
      - num_encoder_layers = 4
      - num_decoder_layers = 4
      - dim_feedforward = 1024
      - dropout = 0.1
    """
    return TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_idx=pad_idx,
        device=device
    ).to(device)


def build_transformer_base(vocab_size: int, device: torch.device,
                           pad_idx: int = 0) -> TransformerSeq2Seq:
    """
    Standard Transformer-Base preset (Vaswani et al.):
      - d_model = 512
      - nhead = 8
      - num_encoder_layers = 6
      - num_decoder_layers = 6
      - dim_feedforward = 2048
      - dropout = 0.1
    """
    return TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=pad_idx,
        device=device
    ).to(device)


def build_vanilla_transformer(vocab_size: int, device: torch.device,
                              pad_idx: int = 0, model_size: str = "base") -> TransformerSeq2Seq:
    """
    Vanilla Transformer Baseline (Vaswani et al., 2017) trained from scratch.
    Satisfies Requirement P2: 'Try vanilla transformer with same data and BPE tokenizer'.
    """
    if model_size == "small":
        return build_transformer_small(vocab_size, device, pad_idx)
    return build_transformer_base(vocab_size, device, pad_idx)

