"""
train.py — Training Pipeline for Gondi GRU + Attention Seq2Seq Translation
==========================================================================
Trains GRU Seq2Seq with Bahdanau Attention from scratch on Gondi dataset.
"""

import os
import math
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from gondi_bpe_tokenizer import GondiTokenizer
from gondi_gru_seq2seq import build_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS           = 30
EMBED_DIM        = 256
HIDDEN_DIM       = 512
NUM_LAYERS       = 2
DROPOUT          = 0.5
TEACHER_FORCING  = 0.5
LR               = 1e-3
TARGET_LANG      = "Hindi"  # "Hindi" or "English"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CSV_PATH         = os.path.join(BASE_DIR, "..", "Datasets", "gondi_training_60k.csv")
TOKENIZER_PATH   = os.path.join(BASE_DIR, "gondi_bpe_model.json")
CHECKPOINT_NAME  = "best_gondi_gru_model.pt" if TARGET_LANG == "Hindi" else "best_gondi_gru_model_eng.pt"
CHECKPOINT_PATH  = os.path.join(BASE_DIR, CHECKPOINT_NAME)

RANDOM_SEED      = 42
torch.manual_seed(RANDOM_SEED)
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  🌿 GRU+Attention Training: Gondi -> {TARGET_LANG}")
print("=" * 70)

tokenizer = GondiTokenizer.load(TOKENIZER_PATH)
PAD_IDX = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
SOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
EOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)
VOCAB_SIZE = tokenizer.vocab_size()

print(f"  Vocab size : {VOCAB_SIZE} | PAD={PAD_IDX}, SOS={SOS_IDX}, EOS={EOS_IDX}")


class GondiDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: GondiTokenizer,
                 src_col: str = "Gondi", tgt_col: str = "Hindi"):
        super().__init__()
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.dropna(subset=[src_col, tgt_col]).reset_index(drop=True)
        self.src_texts = df[src_col].astype(str).tolist()
        self.tgt_texts = df[tgt_col].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_ids = self.tokenizer.encode(self.src_texts[idx])
        tgt_ids = self.tokenizer.encode(self.tgt_texts[idx])
        return (torch.tensor([SOS_IDX] + src_ids + [EOS_IDX], dtype=torch.long),
                torch.tensor([SOS_IDX] + tgt_ids + [EOS_IDX], dtype=torch.long))


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


print(f"  Loading dataset from {CSV_PATH}...")
full_dataset = GondiDataset(CSV_PATH, tokenizer, src_col="Gondi", tgt_col=TARGET_LANG)
total = len(full_dataset)
val_size = int(total * 0.1)
train_size = total - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"  Total samples : {total:,} | Train: {train_size:,} | Val: {val_size:,}")

model = build_model(
    gondi_vocab_size=VOCAB_SIZE,
    target_vocab_size=VOCAB_SIZE,
    device=device,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pad_idx=PAD_IDX
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters : {trainable_params:,} ({trainable_params/1e6:.2f}M)")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

print("\n" + "=" * 70)
print(f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'LR':>10} | {'BLEU':>6}")
print("  " + "─" * 60)

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        src_lengths = (src != PAD_IDX).sum(dim=1)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING, src_lengths=src_lengths)

        out_flat = output[:, 1:].reshape(-1, VOCAB_SIZE)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(out_flat, trg_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0.0
    references, hypotheses = [], []

    with torch.no_grad():
        for i, (src, trg) in enumerate(val_loader):
            src, trg = src.to(device), trg.to(device)
            src_lengths = (src != PAD_IDX).sum(dim=1)

            output = model(src, trg, teacher_forcing_ratio=0.0, src_lengths=src_lengths)
            out_flat = output[:, 1:].reshape(-1, VOCAB_SIZE)
            trg_flat = trg[:, 1:].reshape(-1)

            total_val_loss += criterion(out_flat, trg_flat).item()

            if i < 2:  # Evaluate BLEU on 2 batches for fast logging
                preds = output.argmax(dim=-1)
                for b in range(preds.shape[0]):
                    ref = [tokenizer.id_to_token.get(idx.item(), "") for idx in trg[b] if idx.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                    references.append([ref])

                    p_ids = preds[b].tolist()
                    if EOS_IDX in p_ids:
                        p_ids = p_ids[:p_ids.index(EOS_IDX)]
                    hyp = [tokenizer.id_to_token.get(idx, "") for idx in p_ids if idx not in (PAD_IDX, SOS_IDX)]
                    hypotheses.append(hyp)

    val_loss = total_val_loss / len(val_loader)
    try:
        smoothie = SmoothingFunction().method4
        bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100
    except Exception:
        bleu = 0.0

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)

    print(f"  {epoch:>5} | {train_loss:>11.4f} | {val_loss:>11.4f} | {current_lr:>10.6f} | {bleu:>6.2f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "target_lang": TARGET_LANG,
        }, CHECKPOINT_PATH)
        print(f"         ↳ ✓ Best model saved → {CHECKPOINT_NAME} (val_loss={val_loss:.4f})")

print("\n" + "=" * 70)
print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
print("=" * 70)

