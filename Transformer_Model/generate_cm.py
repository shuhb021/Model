"""
generate_cm.py — Confusion Matrix Generator for Transformer Baselines
=====================================================================
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from gondi_bpe_tokenizer import GondiTokenizer
from gondi_transformer import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = GondiTokenizer.load(os.path.join(base_dir, "gondi_bpe_model.json"))
    VOCAB_SIZE = tokenizer.vocab_size()
    PAD_IDX = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
    SOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
    EOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)

    model = build_model(
        gondi_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        device=device,
        model_size=args.model_size,
        pad_idx=PAD_IDX
    )

    checkpoint_name = f"best_transformer_{args.model_size}_hindi.pt"
    checkpoint_path = os.path.join(base_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv"))
    df = df.dropna(subset=["Gondi", "Hindi"])
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)

    true_tokens, pred_tokens = [], []
    with torch.no_grad():
        for row in df_sample.itertuples():
            src_ids = tokenizer.encode(str(row.Gondi))
            tgt_ids = tokenizer.encode(str(row.Hindi))
            if not src_ids or not tgt_ids:
                continue

            src_tensor = torch.tensor([[SOS_IDX] + src_ids + [EOS_IDX]], dtype=torch.long, device=device)
            pred_ids_batch, _ = model.translate(src_tensor, sos_idx=SOS_IDX, eos_idx=EOS_IDX, max_len=120)

            pred_ids = pred_ids_batch[0].tolist()
            if EOS_IDX in pred_ids:
                pred_ids = pred_ids[:pred_ids.index(EOS_IDX)]
            pred_ids = [tid for tid in pred_ids if tid not in (PAD_IDX, SOS_IDX)]

            min_len = min(len(tgt_ids), len(pred_ids))
            true_tokens.extend(tgt_ids[:min_len])
            pred_tokens.extend(pred_ids[:min_len])

    if not true_tokens:
        print("No tokens collected!")
        return

    top_n = 20
    top_indices = pd.Series(true_tokens).value_counts().head(top_n).index.tolist()
    labels = [tokenizer.id_to_token.get(idx, str(idx)) for idx in top_indices]

    filtered_true = [t for t, p in zip(true_tokens, pred_tokens) if t in top_indices and p in top_indices]
    filtered_pred = [p for t, p in zip(true_tokens, pred_tokens) if t in top_indices and p in top_indices]

    cm = confusion_matrix(filtered_true, filtered_pred, labels=top_indices)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.title(f"Transformer-{args.model_size.capitalize()}: Top-20 Token Confusion Matrix")
    plt.xlabel("Predicted Token")
    plt.ylabel("True Token")
    plt.tight_layout()

    out_png = os.path.join(base_dir, f"transformer_{args.model_size}_confusion_matrix.png")
    plt.savefig(out_png, dpi=300)
    print(f"Saved confusion matrix plot to {out_png}")


if __name__ == "__main__":
    main()

