"""
BiLSTM Seq2Seq with Bahdanau Additive Attention — Gondi Language Translation
=============================================================================
Architecture:
  Encoder : Embedding(vocab, embed_dim) → Bidirectional 2-layer LSTM(hidden_dim)
            Merged fwd+bwd hidden and cell states via linear projection
  Attention: Bahdanau additive attention between decoder hidden state and all encoder outputs
  Decoder : Embedding(vocab, embed_dim) + Context vector → 2-layer LSTM(hidden_dim)
            Concatenation of decoder output and context vector → Linear → Target Vocab
  Teacher Forcing during training; autoregressive greedy/beam decoding for inference.
"""

import random
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional 2-layer LSTM Encoder.
    Returns:
      - encoder_outputs: [batch, src_len, hidden_dim]
      - hidden: [num_layers, batch, hidden_dim]
      - cell:   [num_layers, batch, hidden_dim]
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5,
                 pad_idx: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        # Merge bidirectional representations (hidden_dim * 2 -> hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_outputs = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        src: [batch, src_len]
        src_lengths: [batch]
        """
        embedded = self.dropout(self.embedding(src))  # [B, S, embed_dim]

        if src_lengths is not None:
            # Move lengths to CPU for pack_padded_sequence
            packed = pack_padded_sequence(
                embedded, src_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_out, (hidden, cell) = self.lstm(packed)
            encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # Project merged outputs to hidden_dim
        encoder_outputs = self.fc_outputs(encoder_outputs)  # [B, S, H]

        # Merge forward & backward layers
        # hidden shape: [num_layers * 2, B, H] -> reshape to [num_layers, B, 2 * H]
        hidden = torch.cat([hidden[0::2], hidden[1::2]], dim=2)
        cell = torch.cat([cell[0::2], cell[1::2]], dim=2)

        hidden = torch.tanh(self.fc_hidden(hidden))  # [num_layers, B, H]
        cell = torch.tanh(self.fc_cell(cell))        # [num_layers, B, H]

        return encoder_outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau Additive Attention:
      score(s_{t-1}, h_i) = v^T tanh(W_s * s_{t-1} + W_h * h_i)
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        decoder_hidden:  [batch, hidden_dim] (top layer of decoder)
        encoder_outputs: [batch, src_len, hidden_dim]
        mask:            [batch, src_len] (1 for valid token, 0 for pad)
        returns:
          attn_weights:  [batch, src_len]
        """
        src_len = encoder_outputs.shape[1]
        dec_h = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)  # [B, S, H]

        energy = torch.tanh(self.attn(torch.cat([dec_h, encoder_outputs], dim=2)))  # [B, S, H]
        scores = self.v(energy).squeeze(2)  # [B, S]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        return F.softmax(scores, dim=1)  # [B, S]


class BiLSTMDecoder(nn.Module):
    """
    LSTM Decoder with Attention.
    Inputs: target token, encoder_outputs, previous hidden & cell.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5,
                 pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim)

        # Input to LSTM is concatenated embedding (embed_dim) and context vector (hidden_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Output projection combines decoder hidden output and context vector
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: torch.Tensor, encoder_outputs: torch.Tensor,
                hidden: torch.Tensor, cell: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        token:           [batch]
        encoder_outputs: [batch, src_len, hidden_dim]
        hidden, cell:    [num_layers, batch, hidden_dim]
        """
        token = token.unsqueeze(1)                     # [B, 1]
        embedded = self.dropout(self.embedding(token)) # [B, 1, embed_dim]

        # Bahdanau attention using top-layer decoder hidden state
        attn_w = self.attention(hidden[-1], encoder_outputs, mask=mask)  # [B, S]
        context = torch.bmm(attn_w.unsqueeze(1), encoder_outputs)       # [B, 1, H]

        lstm_in = torch.cat([embedded, context], dim=2)                 # [B, 1, embed_dim + H]
        output, (h, c) = self.lstm(lstm_in, (hidden, cell))             # [B, 1, H]

        out_cat = torch.cat([output.squeeze(1), context.squeeze(1)], dim=1)  # [B, 2*H]
        logits = self.fc_out(out_cat)                                   # [B, vocab_size]

        return logits, h, c, attn_w


class BiLSTMSeq2Seq(nn.Module):
    """
    End-to-end BiLSTM + Attention Seq2Seq Model.
    """
    def __init__(self, encoder: BiLSTMEncoder, decoder: BiLSTMDecoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                src_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src: [batch, src_len]
        trg: [batch, trg_len]
        """
        B, T = src.shape[0], trg.shape[1]
        V = self.decoder.vocab_size
        outputs = torch.zeros(B, T, V, device=self.device)

        mask = (src != self.encoder.pad_idx) if hasattr(self.encoder, "pad_idx") else None
        enc_out, hidden, cell = self.encoder(src, src_lengths)

        dec_input = trg[:, 0]  # First token is <SOS>

        for t in range(1, T):
            logits, hidden, cell, _ = self.decoder(dec_input, enc_out, hidden, cell, mask=mask)
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

        enc_out, hidden, cell = self.encoder(src, src_lengths)
        dec_input = torch.full((B,), sos_idx, dtype=torch.long, device=self.device)

        token_ids = []
        attn_weights = []

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            logits, hidden, cell, attn = self.decoder(dec_input, enc_out, hidden, cell, mask=mask)
            top1 = logits.argmax(dim=1)  # [B]

            token_ids.append(top1)
            attn_weights.append(attn)
            dec_input = top1

            finished = finished | (top1 == eos_idx)
            if finished.all():
                break

        token_ids = torch.stack(token_ids, dim=1)  # [B, out_len]
        return token_ids, attn_weights


def build_bilstm_model(vocab_size: int, device: torch.device,
                       embed_dim: int = 256, hidden_dim: int = 512,
                       num_layers: int = 2, dropout: float = 0.5,
                       pad_idx: int = 0) -> BiLSTMSeq2Seq:
    """Factory helper to instantiate a BiLSTM+Attention model."""
    encoder = BiLSTMEncoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    decoder = BiLSTMDecoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    model = BiLSTMSeq2Seq(encoder, decoder, device).to(device)
    return model

