"""
GRU Seq2Seq with Bahdanau Additive Attention — Gondi Language Translation
==========================================================================
Architecture:
  Encoder : Embedding(vocab, embed_dim=256) → Bidirectional 2-layer GRU(hidden_dim=512)
  Attention: Bahdanau additive attention between decoder hidden state and encoder outputs
  Decoder : Embedding(vocab, embed_dim=256) + Context vector → 2-layer GRU(hidden_dim=512)
            Linear projection [gru_out || context] → vocab_size
  Teacher Forcing during training
"""

import random
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5,
                 pad_idx: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_outputs = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))

        if src_lengths is not None:
            packed = pack_padded_sequence(
                embedded, src_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_out, hidden = self.gru(packed)
            encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            encoder_outputs, hidden = self.gru(embedded)

        encoder_outputs = self.fc_outputs(encoder_outputs)
        hidden = torch.cat([hidden[0::2], hidden[1::2]], dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))

        return encoder_outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_len = encoder_outputs.shape[1]
        dec_h = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        energy = torch.tanh(self.attn(torch.cat([dec_h, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        return F.softmax(scores, dim=1)


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5,
                 pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim)

        self.gru = nn.GRU(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: torch.Tensor, encoder_outputs: torch.Tensor,
                hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = token.unsqueeze(1)
        embedded = self.dropout(self.embedding(token))

        attn_w = self.attention(hidden[-1], encoder_outputs, mask=mask)
        context = torch.bmm(attn_w.unsqueeze(1), encoder_outputs)

        gru_in = torch.cat([embedded, context], dim=2)
        output, h = self.gru(gru_in, hidden)

        out_cat = torch.cat([output.squeeze(1), context.squeeze(1)], dim=1)
        logits = self.fc_out(out_cat)

        return logits, h, attn_w


class GRUSeq2Seq(nn.Module):
    def __init__(self, encoder: GRUEncoder, decoder: GRUDecoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                src_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = src.shape[0], trg.shape[1]
        V = self.decoder.vocab_size
        outputs = torch.zeros(B, T, V, device=self.device)

        mask = (src != self.encoder.pad_idx) if hasattr(self.encoder, "pad_idx") else None
        enc_out, hidden = self.encoder(src, src_lengths)

        dec_input = trg[:, 0]

        for t in range(1, T):
            logits, hidden, _ = self.decoder(dec_input, enc_out, hidden, mask=mask)
            outputs[:, t] = logits

            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if use_teacher else logits.argmax(dim=1)

        return outputs

    @torch.no_grad()
    def translate(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                  max_len: int = 120, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.eval()
        B = src.shape[0]
        mask = (src != self.encoder.pad_idx) if hasattr(self.encoder, "pad_idx") else None

        enc_out, hidden = self.encoder(src, src_lengths)
        dec_input = torch.full((B,), sos_idx, dtype=torch.long, device=self.device)

        token_ids = []
        attn_weights = []
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            logits, hidden, attn = self.decoder(dec_input, enc_out, hidden, mask=mask)
            top1 = logits.argmax(dim=1)

            token_ids.append(top1)
            attn_weights.append(attn)
            dec_input = top1

            finished = finished | (top1 == eos_idx)
            if finished.all():
                break

        token_ids = torch.stack(token_ids, dim=1)
        return token_ids, attn_weights


def build_model(gondi_vocab_size: int, target_vocab_size: int, device: torch.device,
                embed_dim: int = 256, hidden_dim: int = 512,
                num_layers: int = 2, dropout: float = 0.5, pad_idx: int = 0) -> GRUSeq2Seq:
    encoder = GRUEncoder(gondi_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    decoder = GRUDecoder(target_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    return GRUSeq2Seq(encoder, decoder, device).to(device)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(5000, 5000, dev)
    print("GRU+Attention Model initialized successfully!")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

