"""
evaluate.py — Test Benchmark Evaluation for GRU + Attention Gondi Model
=======================================================================
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf

from gondi_bpe_tokenizer import GondiTokenizer
from gondi_gru_seq2seq import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRU+Attention model.")
    parser.add_argument("--checkpoint", type=str, default="best_gondi_gru_model.pt")
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"])
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(base_dir, "gondi_bpe_model.json")
    test_csv = args.test_csv or os.path.join(base_dir, "..", "Datasets", "gondi_test_40k_fixed.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")

    checkpoint_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    tokenizer = GondiTokenizer.load(tokenizer_path)
    PAD_IDX = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
    SOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
    EOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)
    VOCAB_SIZE = tokenizer.vocab_size()

    model = build_model(
        gondi_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        device=device,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.0,
        pad_idx=PAD_IDX
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    df = df.dropna(subset=["Gondi", args.tgt_lang]).reset_index(drop=True)
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.iloc[:args.num_samples]

    print(f"Evaluating GRU+Attention on {len(df):,} samples...")

    references, hypotheses = [], []
    ref_sentences, hyp_sentences = [], []

    with torch.no_grad():
        for row in df.itertuples():
            src_str = str(row.Gondi).strip()
            tgt_str = str(getattr(row, args.tgt_lang)).strip()
            tokens = tokenizer.encode(src_str)
            if not tokens:
                continue

            src_tensor = torch.tensor([[SOS_IDX] + tokens + [EOS_IDX]], dtype=torch.long, device=device)
            src_lengths = torch.tensor([len(tokens) + 2], dtype=torch.long, device=device)

            pred_tokens, _ = model.translate(src_tensor, sos_idx=SOS_IDX, eos_idx=EOS_IDX, max_len=100, src_lengths=src_lengths)
            pred_ids = pred_tokens[0].tolist()

            words = []
            for tid in pred_ids:
                if tid == EOS_IDX:
                    break
                if tid not in (PAD_IDX, SOS_IDX):
                    w = tokenizer.id_to_token.get(tid, "")
                    if w:
                        words.append(w)
            hyp_str = "".join(words).replace(" ", " ").strip()

            ref_sentences.append(tgt_str)
            hyp_sentences.append(hyp_str)
            references.append([tgt_str.split()])
            hypotheses.append(hyp_str.split())

    smoothie = SmoothingFunction().method4
    bleu_4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100
    try:
        chrf = corpus_chrf(ref_sentences, hyp_sentences) * 100
    except Exception:
        chrf = 0.0

    print("=" * 60)
    print(f"  BLEU-4:  {bleu_4:.2f}")
    print(f"  chrF++:  {chrf:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

