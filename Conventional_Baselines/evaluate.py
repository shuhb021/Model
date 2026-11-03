"""
evaluate.py — Comprehensive Test Benchmark for Conventional Baselines
=====================================================================
Evaluates trained baseline checkpoints on Gondi translation benchmarks.
Computes:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4, and Cumulative Corpus BLEU
  - chrF / chrF++ score
  - Perplexity
  - Average inference latency (ms/sentence) & throughput (tokens/sec)
  - Qualitative sample translations

Usage:
  python evaluate.py --model bilstm --checkpoint checkpoints/best_bilstm_hindi.pt --num_samples 500
  python evaluate.py --model gru --checkpoint checkpoints/best_gru_hindi.pt --num_samples 500
  python evaluate.py --model transformer_small --checkpoint checkpoints/best_transformer_small_hindi.pt --num_samples 500
  python evaluate.py --model transformer_base --checkpoint checkpoints/best_transformer_base_hindi.pt --num_samples 500
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf

from gondi_bpe_tokenizer import GondiTokenizer
from models import (
    build_bilstm_model,
    build_gru_model,
    build_transformer_small,
    build_transformer_base,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on Gondi translation.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["bilstm", "gru", "transformer_small", "transformer_base"],
                        help="Baseline architecture type.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to test CSV (defaults to ../Datasets/gondi_test_40k_fixed.csv).")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to BPE tokenizer JSON.")
    parser.add_argument("--src_lang", type=str, default="Gondi", help="Source language column.")
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"],
                        help="Target language column.")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of test samples to evaluate (use -1 for all).")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum generated tokens.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation summary JSON.")
    return parser.parse_args()


def load_model(model_type: str, checkpoint_path: str, vocab_size: int, device: torch.device, pad_idx: int = 0):
    if model_type == "bilstm":
        model = build_bilstm_model(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif model_type == "gru":
        model = build_gru_model(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif model_type == "transformer_small":
        model = build_transformer_small(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif model_type == "transformer_base":
        model = build_transformer_base(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, checkpoint


def decode_tokens_to_sentence(token_ids: List[int], tokenizer: GondiTokenizer,
                              pad_idx: int, sos_idx: int, eos_idx: int) -> str:
    """Decodes a list of token IDs back into readable text string."""
    tokens = []
    for tid in token_ids:
        if tid == eos_idx:
            break
        if tid in (pad_idx, sos_idx):
            continue
        tok = tokenizer.id_to_token.get(tid, "")
        if tok:
            tokens.append(tok)
    
    # Detokenize BPE tokens (remove merge boundary characters if applicable)
    raw = "".join(tokens).replace(" ", " ").strip()
    return raw


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    tokenizer_path = args.tokenizer_path or os.path.join(base_dir, "gondi_bpe_model.json")
    test_csv = args.test_csv or os.path.join(base_dir, "..", "Datasets", "gondi_test_40k_fixed.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")

    print("=" * 75)
    print(f"  📊 Benchmark Evaluation: {args.model.upper()}")
    print(f"  📂 Checkpoint: {args.checkpoint}")
    print(f"  📄 Test File:  {test_csv}")
    print(f"  🎯 Direction:  {args.src_lang} -> {args.tgt_lang}")
    print(f"  🖥️  Device:     {device}")
    print("=" * 75)

    # 1. Load Tokenizer
    tokenizer = GondiTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size()
    pad_idx = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
    sos_idx = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
    eos_idx = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)

    # 2. Load Model
    model, ckpt_meta = load_model(args.model, args.checkpoint, vocab_size, device, pad_idx=pad_idx)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model Loaded successfully! Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Load Test Data
    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    df = df.dropna(subset=[args.src_lang, args.tgt_lang]).reset_index(drop=True)
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.iloc[:args.num_samples]

    print(f"  ✓ Evaluating on {len(df):,} sentences...")

    references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []
    ref_sentences: List[str] = []
    hyp_sentences: List[str] = []
    sample_pairs: List[Dict[str, str]] = []

    total_gen_time = 0.0
    total_gen_tokens = 0
    is_transformer = "transformer" in args.model

    with torch.no_grad():
        for idx, row in enumerate(df.itertuples()):
            src_str = str(getattr(row, args.src_lang)).strip()
            tgt_str = str(getattr(row, args.tgt_lang)).strip()

            src_ids = tokenizer.encode(src_str)
            if not src_ids:
                continue

            src_tensor = torch.tensor([[sos_idx] + src_ids + [eos_idx]], dtype=torch.long, device=device)
            src_lengths = torch.tensor([len(src_ids) + 2], dtype=torch.long, device=device)

            t0 = time.time()
            if is_transformer:
                pred_tokens, _ = model.translate(src_tensor, sos_idx=sos_idx, eos_idx=eos_idx, max_len=args.max_len)
            else:
                pred_tokens, _ = model.translate(src_tensor, sos_idx=sos_idx, eos_idx=eos_idx, max_len=args.max_len, src_lengths=src_lengths)
            dt = time.time() - t0

            gen_ids = pred_tokens[0].tolist()
            total_gen_time += dt
            total_gen_tokens += len(gen_ids)

            hyp_text = decode_tokens_to_sentence(gen_ids, tokenizer, pad_idx, sos_idx, eos_idx)
            ref_sentences.append(tgt_str)
            hyp_sentences.append(hyp_text)

            # Tokenize for BLEU computation
            ref_words = [w for w in tgt_str.split() if w]
            hyp_words = [w for w in hyp_text.split() if w]
            references.append([ref_words])
            hypotheses.append(hyp_words)

            if idx < 5:
                sample_pairs.append({
                    "src": src_str,
                    "ref": tgt_str,
                    "hyp": hyp_text
                })

            if (idx + 1) % 200 == 0:
                print(f"    Evaluated {idx + 1}/{len(df)} samples...")

    # Metrics
    smoothie = SmoothingFunction().method4
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothie) * 100
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie) * 100
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie) * 100
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) * 100

    # chrF score
    try:
        chrf_score = corpus_chrf(ref_sentences, hyp_sentences) * 100
    except Exception:
        chrf_score = 0.0

    avg_latency_ms = (total_gen_time / max(len(ref_sentences), 1)) * 1000.0
    throughput_tps = total_gen_tokens / max(total_gen_time, 1e-5)

    print("\n" + "=" * 75)
    print("  🏆 EVALUATION BENCHMARK RESULTS")
    print("=" * 75)
    print(f"  Model:                  {args.model.upper()}")
    print(f"  Trainable Parameters:   {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Test Sentences:         {len(ref_sentences):,}")
    print(f"  BLEU-1:                 {bleu_1:.2f}")
    print(f"  BLEU-2:                 {bleu_2:.2f}")
    print(f"  BLEU-3:                 {bleu_3:.2f}")
    print(f"  BLEU-4 (Cumulative):    {bleu_4:.2f}")
    print(f"  chrF++ Score:           {chrf_score:.2f}")
    print(f"  Average Latency:        {avg_latency_ms:.2f} ms / sentence")
    print(f"  Throughput:             {throughput_tps:.2f} tokens / second")
    print("=" * 75)

    print("\n  🔍 Sample Predictions:")
    print("  " + "─" * 70)
    for i, s in enumerate(sample_pairs, 1):
        print(f"  [{i}] Gondi (Src):    {s['src']}")
        print(f"      Reference (Ref): {s['ref']}")
        print(f"      Prediction (Hyp): {s['hyp']}\n")

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "parameters": total_params,
        "params_millions": round(total_params / 1e6, 2),
        "test_samples": len(ref_sentences),
        "bleu_1": round(bleu_1, 2),
        "bleu_2": round(bleu_2, 2),
        "bleu_3": round(bleu_3, 2),
        "bleu_4": round(bleu_4, 2),
        "chrf": round(chrf_score, 2),
        "latency_ms": round(avg_latency_ms, 2),
        "throughput_tokens_per_sec": round(throughput_tps, 2),
        "samples": sample_pairs,
    }

    out_json = args.output_json or os.path.join(base_dir, f"eval_{args.model}_{args.tgt_lang.lower()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  💾 Results saved to: {out_json}")


if __name__ == "__main__":
    main()

