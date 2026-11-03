"""
LSTM Seq2Seq with Attention — Gondi Language Translation
Architecture matches the diagram exactly:
  Encoder : Embedding(256) → Bidirectional 2-layer LSTM(hidden=512) → all hidden states
  Attention: encoder hidden states + decoder hidden state → context vector
  Decoder : Embedding(256) → 2-layer LSTM(hidden=512, input=embed+context) → Linear → tokens
  Teacher Forcing during training
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ─────────────────────────────────────────────
#  ENCODER
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Gondi sentence tokens → Embedding(256) → Bidirectional 2-layer LSTM(512)
    Returns ALL hidden states (used by attention) + final (hidden, cell).
    Bidirectional outputs are merged via learned linear projections.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True       # ← bidirectional encoder
        )

        # Bidirectional merge: project concatenated fwd+bwd (2*H) → H
        self.fc_hidden  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell    = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_outputs = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor = None):
        """
        src         : [batch, src_len]  (Gondi token IDs)
        src_lengths : [batch]           (actual lengths, excluding padding)
        ──────────────────────────────────────────────────────
        returns
          encoder_outputs : [batch, src_len, hidden_dim]   ← arrow ①
          hidden          : [num_layers, batch, hidden_dim]
          cell            : [num_layers, batch, hidden_dim]
        """
        embedded = self.dropout(self.embedding(src))          # [B, S, 256]

        # ── Pack padded sequences for efficient computation ──
        if src_lengths is not None:
            packed = pack_padded_sequence(
                embedded, src_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_out, (hidden, cell) = self.lstm(packed)
            # Unpack back to padded tensor
            encoder_outputs, _ = pad_packed_sequence(
                packed_out, batch_first=True
            )   # [B, S, 2*H]
        else:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)  # [B, S, 2*H]

        # ── Merge bidirectional outputs: concat fwd+bwd → project to H ──
        encoder_outputs = self.fc_outputs(encoder_outputs)    # [B, S, H]

        # ── Merge bidirectional hidden & cell states ─────────
        # hidden shape from bidir LSTM: [num_layers*2, B, H]
        # Reshape to [num_layers, B, 2*H] by concatenating fwd & bwd
        hidden = torch.cat(
            [hidden[0::2], hidden[1::2]], dim=2
        )  # [num_layers, B, 2*H]
        cell = torch.cat(
            [cell[0::2], cell[1::2]], dim=2
        )  # [num_layers, B, 2*H]

        # Project merged states back to hidden_dim for decoder compatibility
        hidden = torch.tanh(self.fc_hidden(hidden))           # [num_layers, B, H]
        cell   = torch.tanh(self.fc_cell(cell))               # [num_layers, B, H]

        return encoder_outputs, hidden, cell


# ─────────────────────────────────────────────
#  ATTENTION MECHANISM
# ─────────────────────────────────────────────
class Attention(nn.Module):
    """
    Bahdanau-style additive attention.
    Inputs : top-layer decoder hidden state + all encoder hidden states
    Output : context vector (arrow ③ in diagram)
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        # Linear 512 label in diagram = project concatenated [dec_h ; enc_out] → energy
        self.attn  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v     = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        decoder_hidden  : [batch, hidden_dim]   (top LSTM layer)
        encoder_outputs : [batch, src_len, hidden_dim]
        returns
          attn_weights : [batch, src_len]
        """
        src_len = encoder_outputs.shape[1]
        dec_h   = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)  # [B, S, H]

        energy  = torch.tanh(self.attn(torch.cat([dec_h, encoder_outputs], dim=2)))
        scores  = self.v(energy).squeeze(2)           # [B, S]
        return F.softmax(scores, dim=1)               # [B, S]


# ─────────────────────────────────────────────
#  DECODER
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    """
    One step of decoding:
      token → Embedding(256)
      attention(dec_hidden, enc_outputs) → context vector  (arrow ③)
      [embed ‖ context] → 2-layer LSTM(512)                 (Linear 512 label)
      [lstm_out ‖ context] → Linear → vocab logits           (arrow ①→② output)
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention  = Attention(hidden_dim)

        # LSTM input = embed(256) + context(512)  →  "Linear 512" shown in diagram
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Final projection to vocabulary (arrow ① decoder LSTM → output tokens, arrow ②)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)   # [lstm_out ‖ context]
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: torch.Tensor,
                encoder_outputs: torch.Tensor,
                hidden: torch.Tensor,
                cell: torch.Tensor):
        """
        token           : [batch]               (current input token)
        encoder_outputs : [batch, src_len, 512]
        hidden / cell   : [num_layers, batch, 512]
        ──────────────────────────────────────────────────────
        returns
          logits       : [batch, vocab_size]
          hidden       : [num_layers, batch, 512]
          cell         : [num_layers, batch, 512]
          attn_weights : [batch, src_len]
        """
        token    = token.unsqueeze(1)                          # [B, 1]
        embedded = self.dropout(self.embedding(token))         # [B, 1, 256]

        # ── Attention → context vector (arrow ③) ──────────────
        attn_w  = self.attention(hidden[-1], encoder_outputs)  # [B, S]
        context = torch.bmm(attn_w.unsqueeze(1), encoder_outputs)  # [B, 1, 512]

        # ── LSTM step (Linear 512 in diagram) ─────────────────
        lstm_in         = torch.cat([embedded, context], dim=2)  # [B, 1, 768]
        output, (h, c)  = self.lstm(lstm_in, (hidden, cell))     # [B, 1, 512]

        # ── Output projection (arrow ①→② decoder side) ────────
        out_cat = torch.cat([output.squeeze(1), context.squeeze(1)], dim=1)  # [B, 1024]
        logits  = self.fc_out(out_cat)                           # [B, vocab_size]

        return logits, h, c, attn_w


# ─────────────────────────────────────────────
#  SEQ2SEQ WRAPPER
# ─────────────────────────────────────────────
class Seq2SeqWithAttention(nn.Module):
    """
    Full Encoder-Decoder pipeline with Teacher Forcing.
    Compatible with any external tokenizer — just pass token ID tensors.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    # ── Training forward ──────────────────────────────────────
    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                src_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        src         : [batch, src_len]   Gondi token IDs
        trg         : [batch, trg_len]   Target token IDs (English / Hindi)
        teacher_forcing_ratio : probability of using ground-truth token as next input
        src_lengths : [batch]            actual source lengths (for packed sequences)

        returns
          outputs : [batch, trg_len, tgt_vocab_size]
        """
        B, T       = src.shape[0], trg.shape[1]
        V          = self.decoder.vocab_size
        outputs    = torch.zeros(B, T, V).to(self.device)

        # ── Encode (arrow ① Encoder → all hidden states) ──────
        enc_out, hidden, cell = self.encoder(src, src_lengths)

        # First decoder input = <SOS> token
        dec_input = trg[:, 0]

        for t in range(1, T):
            logits, hidden, cell, _ = self.decoder(dec_input, enc_out, hidden, cell)
            outputs[:, t] = logits

            # Teacher Forcing (diagram label: "Teacher forcing")
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input   = trg[:, t] if use_teacher else logits.argmax(1)

        return outputs   # [B, T, V]

    # ── Inference (no teacher forcing) ───────────────────────
    @torch.no_grad()
    def translate(self, src: torch.Tensor,
                  sos_idx: int, eos_idx: int,
                  max_len: int = 100,
                  src_lengths: torch.Tensor = None):
        """
        src         : [batch, src_len]
        sos_idx     : <SOS> index from YOUR tokenizer
        eos_idx     : <EOS> index from YOUR tokenizer
        src_lengths : [batch]  actual source lengths (for packed sequences)

        returns
          token_ids      : [batch, out_len]   generated token IDs
          attn_weights   : list of [batch, src_len]  per step
        """
        self.eval()
        B = src.shape[0]

        enc_out, hidden, cell = self.encoder(src, src_lengths)
        dec_input  = torch.full((B,), sos_idx, dtype=torch.long, device=self.device)

        token_ids   = []
        attn_weights = []

        for _ in range(max_len):
            logits, hidden, cell, attn = self.decoder(dec_input, enc_out, hidden, cell)
            top1 = logits.argmax(1)              # [B]
            token_ids.append(top1)
            attn_weights.append(attn)
            dec_input = top1

            if (top1 == eos_idx).all():
                break

        token_ids = torch.stack(token_ids, dim=1)   # [B, out_len]
        return token_ids, attn_weights


# ─────────────────────────────────────────────
#  FACTORY — plug in YOUR tokenizer vocab sizes
# ─────────────────────────────────────────────
def build_model(gondi_vocab_size: int,
                target_vocab_size: int,
                device: torch.device,
                embed_dim:   int   = 256,
                hidden_dim:  int   = 512,
                num_layers:  int   = 2,
                dropout:     float = 0.5) -> Seq2SeqWithAttention:
    """
    gondi_vocab_size  : len(gondi_tokenizer.vocab)
    target_vocab_size : len(target_tokenizer.vocab)   (English or Hindi)
    """
    encoder = Encoder(gondi_vocab_size,  embed_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(target_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    model   = Seq2SeqWithAttention(encoder, decoder, device).to(device)
    return model


# ─────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion,
                    grad_clip: float = 1.0,
                    teacher_forcing_ratio: float = 0.5,
                    pad_idx: int = 0) -> float:
    """
    One full pass over the dataloader.
    criterion should use ignore_index = your PAD token index.
    """
    model.train()
    total_loss = 0.0

    for src, trg in dataloader:
        src, trg = src.to(model.device), trg.to(model.device)

        # Compute actual source lengths (non-pad tokens) for packed sequences
        src_lengths = (src != pad_idx).sum(dim=1)             # [B]

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio,
                       src_lengths=src_lengths)               # [B, T, V]

        # Flatten: skip the first token (SOS) for loss
        out_flat = output[:, 1:].reshape(-1, model.decoder.vocab_size)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(out_flat, trg_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, pad_idx: int = 0) -> float:
    model.eval()
    total_loss = 0.0

    for src, trg in dataloader:
        src, trg = src.to(model.device), trg.to(model.device)
        src_lengths = (src != pad_idx).sum(dim=1)             # [B]
        output   = model(src, trg, teacher_forcing_ratio=0.0,
                         src_lengths=src_lengths)
        out_flat = output[:, 1:].reshape(-1, model.decoder.vocab_size)
        trg_flat = trg[:, 1:].reshape(-1)
        total_loss += criterion(out_flat, trg_flat).item()

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Plug your custom tokenizer vocab sizes here ────────────
    GONDI_VOCAB_SIZE  = 5000   # ← replace with len(your_gondi_tokenizer.vocab)
    TARGET_VOCAB_SIZE = 10000  # ← replace with len(your_target_tokenizer.vocab)
    PAD_IDX           = 0      # ← replace with your tokenizer's PAD index
    SOS_IDX           = 1      # ← replace with your tokenizer's SOS index
    EOS_IDX           = 2      # ← replace with your tokenizer's EOS index

    model = build_model(
        gondi_vocab_size  = GONDI_VOCAB_SIZE,
        target_vocab_size = TARGET_VOCAB_SIZE,
        device            = device,
        embed_dim         = 256,
        hidden_dim        = 512,
        num_layers        = 2,
        dropout           = 0.5,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}")
    print(model)

    # ── Dummy forward pass ─────────────────────────────────────
    B, SRC_LEN, TRG_LEN = 4, 20, 18
    src_dummy = torch.randint(3, GONDI_VOCAB_SIZE,  (B, SRC_LEN)).to(device)
    trg_dummy = torch.randint(3, TARGET_VOCAB_SIZE, (B, TRG_LEN)).to(device)

    # Compute source lengths for packed sequences
    src_lengths = (src_dummy != PAD_IDX).sum(dim=1)   # [B]

    out = model(src_dummy, trg_dummy, teacher_forcing_ratio=0.5,
                src_lengths=src_lengths)
    print(f"\nForward pass OK  →  output shape: {out.shape}")   # [4, 18, 10000]

    # ── Dummy translate ────────────────────────────────────────
    tokens, attns = model.translate(src_dummy, sos_idx=SOS_IDX,
                                    eos_idx=EOS_IDX, max_len=30,
                                    src_lengths=src_lengths)
    print(f"Translate OK     →  tokens shape : {tokens.shape}")

    # ── Optimizer & Loss (ready to train) ─────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print("\n✅ Model ready. Plug in your DataLoader and start training with train_one_epoch().")
