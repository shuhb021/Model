"""
train.py — Training Pipeline for Standard Transformer Seq2Seq Baselines
========================================================================
Trains Transformer-Small or Transformer-Base from scratch on Gondi dataset.
"""

import os
import math
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from gondi_bpe_tokenizer import GondiTokenizer
from gondi_transformer import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Baseline on Gondi.")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base"],
                        help="Transformer size preset.")
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


args = parse_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "Datasets", "gondi_training_60k.csv")
TOKENIZER_PATH = os.path.join(BASE_DIR, "gondi_bpe_model.json")
CHECKPOINT_NAME = f"best_transformer_{args.model_size}_{args.tgt_lang.lower()}.pt"
CHECKPOINT_PATH = os.path.join(BASE_DIR, CHECKPOINT_NAME)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = args.lr or (5e-4 if args.model_size == "small" else 3e-4)

print("=" * 70)
print(f"  ⚡ Standard Transformer-{args.model_size.capitalize()} Training")
print(f"  🎯 Direction: Gondi -> {args.tgt_lang}")
print("=" * 70)

tokenizer = GondiTokenizer.load(TOKENIZER_PATH)
PAD_IDX = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
SOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
EOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)
VOCAB_SIZE = tokenizer.vocab_size()

print(f"  Vocab size : {VOCAB_SIZE} | PAD={PAD_IDX}, SOS={SOS_IDX}, EOS={EOS_IDX}")


class GondiDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: GondiTokenizer, src_col: str, tgt_col: str):
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
full_dataset = GondiDataset(CSV_PATH, tokenizer, "Gondi", args.tgt_lang)
total = len(full_dataset)
val_size = int(total * 0.1)
train_size = total - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

print(f"  Total samples : {total:,} | Train: {train_size:,} | Val: {val_size:,}")

model = build_model(
    gondi_vocab_size=VOCAB_SIZE,
    target_vocab_size=VOCAB_SIZE,
    device=device,
    model_size=args.model_size,
    pad_idx=PAD_IDX
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters : {trainable_params:,} ({trainable_params/1e6:.2f}M)")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

print("\n" + "=" * 70)
print(f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'LR':>10} | {'BLEU':>6}")
print("  " + "─" * 60)

best_val_loss = float("inf")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_train_loss = 0.0

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        trg_in = trg[:, :-1]
        trg_expected = trg[:, 1:]

        optimizer.zero_grad()
        output = model(src, trg_in)

        out_flat = output.reshape(-1, VOCAB_SIZE)
        trg_flat = trg_expected.reshape(-1)

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
            trg_in = trg[:, :-1]
            trg_expected = trg[:, 1:]

            output = model(src, trg_in)
            out_flat = output.reshape(-1, VOCAB_SIZE)
            trg_flat = trg_expected.reshape(-1)

            total_val_loss += criterion(out_flat, trg_flat).item()

            if i < 2:
                trans_tokens, _ = model.translate(src, sos_idx=SOS_IDX, eos_idx=EOS_IDX, max_len=60)
                for b in range(src.size(0)):
                    ref = [tokenizer.id_to_token.get(idx.item(), "") for idx in trg[b] if idx.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                    references.append([ref])

                    p_ids = trans_tokens[b].tolist()
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
            "model_size": args.model_size,
            "epoch": epoch,
            "val_loss": val_loss,
            "target_lang": args.tgt_lang,
        }, CHECKPOINT_PATH)
        print(f"         ↳ ✓ Best model saved → {CHECKPOINT_NAME} (val_loss={val_loss:.4f})")

print("\n" + "=" * 70)
print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
print("=" * 70)

