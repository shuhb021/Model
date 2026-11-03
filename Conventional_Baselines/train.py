"""
train.py — Unified Training Pipeline for Conventional Seq2Seq Baselines
========================================================================
Supports training from scratch on the Gondi parallel corpus:
  1. BiLSTM + Bahdanau Attention
  2. GRU + Bahdanau Attention
  3. Standard Transformer-Small
  4. Standard Transformer-Base

Usage:
  python train.py --model bilstm --tgt_lang Hindi --epochs 30
  python train.py --model gru --tgt_lang English --epochs 30
  python train.py --model transformer_small --tgt_lang Hindi --epochs 30
  python train.py --model transformer_base --tgt_lang English --epochs 30
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import GondiTranslationDataset, pad_collate_fn, load_unified_tokenizer
from models import (
    build_bilstm_model,
    build_gru_model,
    build_transformer_small,
    build_transformer_base,
    build_vanilla_transformer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train conventional seq2seq baselines on Gondi corpus.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["bilstm", "gru", "transformer_small", "transformer_base", "vanilla_transformer"],
                        help="Baseline architecture to train.")
    parser.add_argument("--src_lang", type=str, default="Gondi", help="Source column name.")
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"],
                        help="Target language column.")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to training CSV. Defaults to ../Datasets/gondi_training_60k.csv.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to BPE tokenizer JSON.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-3 for RNN, 5e-4 for Transformer).")
    parser.add_argument("--teacher_forcing", type=float, default=0.5, help="Teacher forcing ratio for RNNs.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Cross entropy label smoothing.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional subsample limit for fast smoke testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints.")
    return parser.parse_args()


def train_one_epoch_rnn(model, dataloader, optimizer, criterion, grad_clip, teacher_forcing, pad_idx, device):
    model.train()
    total_loss = 0.0

    for src, trg, src_lengths in dataloader:
        src, trg = src.to(device), trg.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing, src_lengths=src_lengths)
        # output: [B, T, V]
        # Ignore initial <SOS> token at index 0 for loss computation
        out_flat = output[:, 1:].reshape(-1, output.shape[-1])
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(out_flat, trg_flat)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_rnn(model, dataloader, criterion, pad_idx, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, trg, src_lengths in dataloader:
            src, trg = src.to(device), trg.to(device)
            src_lengths = src_lengths.to(device)

            output = model(src, trg, teacher_forcing_ratio=0.0, src_lengths=src_lengths)
            out_flat = output[:, 1:].reshape(-1, output.shape[-1])
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(out_flat, trg_flat)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_one_epoch_transformer(model, dataloader, optimizer, criterion, grad_clip, pad_idx, device):
    model.train()
    total_loss = 0.0

    for src, trg, _ in dataloader:
        src, trg = src.to(device), trg.to(device)
        trg_input = trg[:, :-1]
        trg_expected = trg[:, 1:]

        optimizer.zero_grad()
        output = model(src, trg_input)
        # output: [B, T-1, V]
        out_flat = output.reshape(-1, output.shape[-1])
        trg_flat = trg_expected.reshape(-1)

        loss = criterion(out_flat, trg_flat)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_transformer(model, dataloader, criterion, pad_idx, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, trg, _ in dataloader:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:]

            output = model(src, trg_input)
            out_flat = output.reshape(-1, output.shape[-1])
            trg_flat = trg_expected.reshape(-1)

            loss = criterion(out_flat, trg_flat)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_sample_bleu(model, is_transformer: bool, val_loader, tokenizer, pad_idx, sos_idx, eos_idx, device, num_batches: int = 2):
    """Computes BLEU on sample validation batches to monitor translation quality per epoch."""
    model.eval()
    references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []

    batches_evaluated = 0
    with torch.no_grad():
        for src, trg, src_lengths in val_loader:
            src, trg = src.to(device), trg.to(device)
            src_lengths = src_lengths.to(device)

            if is_transformer:
                trans_tokens, _ = model.translate(src, sos_idx=sos_idx, eos_idx=eos_idx, max_len=60)
            else:
                trans_tokens, _ = model.translate(src, sos_idx=sos_idx, eos_idx=eos_idx, max_len=60, src_lengths=src_lengths)

            for i in range(src.size(0)):
                # Reference tokens excluding PAD, SOS, EOS
                ref_ids = [idx.item() for idx in trg[i] if idx.item() not in (pad_idx, sos_idx, eos_idx)]
                ref_words = [tokenizer.id_to_token.get(idx, "") for idx in ref_ids]
                references.append([ref_words])

                # Hypothesis tokens
                hyp_ids = trans_tokens[i].tolist()
                if eos_idx in hyp_ids:
                    hyp_ids = hyp_ids[:hyp_ids.index(eos_idx)]
                hyp_ids = [idx for idx in hyp_ids if idx not in (pad_idx, sos_idx)]
                hyp_words = [tokenizer.id_to_token.get(idx, "") for idx in hyp_ids]
                hypotheses.append(hyp_words)

            batches_evaluated += 1
            if batches_evaluated >= num_batches:
                break

    smoothie = SmoothingFunction().method4
    try:
        score = corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100.0
    except Exception:
        score = 0.0
    return score


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Default paths
    csv_path = args.csv_path or os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")
    tokenizer_path = args.tokenizer_path or os.path.join(base_dir, "gondi_bpe_model.json")
    save_dir = args.save_dir or os.path.join(base_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 75)
    print(f"  🚀 Baseline Model Training: {args.model.upper()}")
    print(f"  🌍 Task: Gondi -> {args.tgt_lang}")
    print(f"  🖥️  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 75)

    # 1. Load Tokenizer (supports both native BPE model and HuggingFace tokenizer.json)
    tokenizer = load_unified_tokenizer(tokenizer_path)
    vocab_size = tokenizer.vocab_size()
    pad_idx = tokenizer.pad_idx
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx

    print(f"  ✓ Tokenizer: {vocab_size} tokens | PAD={pad_idx}, SOS={sos_idx}, EOS={eos_idx}")

    # 2. Prepare Dataset
    print(f"  📂 Loading data from: {csv_path}")
    dataset = GondiTranslationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        src_col=args.src_lang,
        tgt_col=args.tgt_lang,
        max_samples=args.max_samples
    )
    total_samples = len(dataset)
    val_size = int(total_samples * args.val_split)
    train_size = total_samples - val_size

    train_data, val_data = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    collate_fn = lambda b: pad_collate_fn(b, pad_idx=pad_idx)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available())

    print(f"  ✓ Total: {total_samples:,} | Train: {train_size:,} | Val: {val_size:,}")
    print(f"  ✓ Batches per epoch: {len(train_loader)} train | {len(val_loader)} val")

    # 3. Instantiate Model
    is_transformer = "transformer" in args.model

    if args.model == "bilstm":
        dropout = args.dropout if args.dropout is not None else 0.5
        lr = args.lr or 1e-3
        model = build_bilstm_model(vocab_size=vocab_size, device=device, dropout=dropout, pad_idx=pad_idx)
    elif args.model == "gru":
        dropout = args.dropout if args.dropout is not None else 0.5
        lr = args.lr or 1e-3
        model = build_gru_model(vocab_size=vocab_size, device=device, dropout=dropout, pad_idx=pad_idx)
    elif args.model == "transformer_small":
        lr = args.lr or 5e-4
        model = build_transformer_small(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif args.model in ("transformer_base", "vanilla_transformer"):
        lr = args.lr or 3e-4
        model = build_vanilla_transformer(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Architecture: {model.__class__.__name__}")
    print(f"  ✓ Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 4. Optimization & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    checkpoint_filename = f"best_{args.model}_{args.tgt_lang.lower()}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_filename)
    history_path = os.path.join(save_dir, f"history_{args.model}_{args.tgt_lang.lower()}.json")

    # 5. Training Loop
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 75)
    print(f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'Perplexity':>11} | {'Val BLEU':>9} | {'LR':>9} | {'Time':>6}")
    print("  " + "─" * 70)

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        if is_transformer:
            train_loss = train_one_epoch_transformer(
                model, train_loader, optimizer, criterion, args.grad_clip, pad_idx, device
            )
            val_loss = evaluate_transformer(
                model, val_loader, criterion, pad_idx, device
            )
        else:
            train_loss = train_one_epoch_rnn(
                model, train_loader, optimizer, criterion, args.grad_clip,
                args.teacher_forcing, pad_idx, device
            )
            val_loss = evaluate_rnn(
                model, val_loader, criterion, pad_idx, device
            )

        elapsed = time.time() - start_t
        ppl = math.exp(min(val_loss, 20.0))

        # Sample BLEU
        val_bleu = compute_sample_bleu(
            model, is_transformer, val_loader, tokenizer, pad_idx, sos_idx, eos_idx, device, num_batches=2
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        print(f"  {epoch:>5} | {train_loss:>11.4f} | {val_loss:>11.4f} | {ppl:>11.2f} | {val_bleu:>9.2f} | {current_lr:>9.2e} | {elapsed:>5.1f}s")

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": ppl,
            "val_bleu": val_bleu,
            "lr": current_lr,
            "time_seconds": elapsed,
        }
        history.append(epoch_stats)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_name": args.model,
                "model_class": model.__class__.__name__,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "src_lang": args.src_lang,
                "tgt_lang": args.tgt_lang,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "parameters": trainable_params,
            }, checkpoint_path)
            print(f"         ↳ ✓ Best checkpoint saved → {checkpoint_filename} (Val Loss: {val_loss:.4f})")

    # Save training history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=" * 75)
    print(f"  🎉 Training Finished! Best Validation Loss: {best_val_loss:.4f}")
    print(f"  💾 Saved to: {checkpoint_path}")
    print(f"  📊 Training log saved to: {history_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

