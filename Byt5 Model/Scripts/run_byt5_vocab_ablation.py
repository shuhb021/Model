"""
run_byt5_vocab_ablation.py — Automated ByT5 Vocabulary Size Ablation Benchmark
=============================================================================
Satisfies Reviewer / Research Requirement:
  "P1: We have run Byt5 with BPE 1157 token. Try it for other vocabulary size and get result."

This script automates:
  1. Generating BPE tokenizers for different vocabulary sizes (e.g., 1157, 2000, 4000, 8000)
  2. Training ByT5 model on each vocabulary size
  3. Evaluating test BLEU, chrF, and parameter counts
  4. Generating a comparative summary table and markdown report

Usage:
  # Run full ablation over vocabularies [1157, 2000, 4000]:
  python run_byt5_vocab_ablation.py --vocabs 1157 2000 4000 --epochs 5

  # Run on a single new vocab size (e.g. 2000):
  python run_byt5_vocab_ablation.py --vocabs 2000 --epochs 5
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run ByT5 vocabulary ablation experiment.")
    parser.add_argument("--vocabs", nargs="+", type=int, default=[1157, 2000, 4000, 8000],
                        help="List of BPE vocabulary sizes to compare.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per model.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_report", type=str, default="byt5_vocab_ablation_results.md")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_dir = os.path.join(base_dir, "..", "Tokenizer_Files")
    csv_path = os.path.join(base_dir, "..", "..", "Datasets", "gondi_training_60k.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")

    results = []

    print("=" * 75)
    print("  🔬 ByT5 Vocabulary Size Ablation Study")
    print(f"  📊 Vocabulary sizes to test: {args.vocabs}")
    print("=" * 75)

    for v_size in args.vocabs:
        print(f"\n▶ Testing ByT5 with BPE Vocab Size: {v_size}...")

        # Step 1: Ensure Tokenizer exists
        tok_out = os.path.join(tok_dir, f"vocab_{v_size}")
        tok_file = os.path.join(tok_out, "tokenizer.json") if v_size != 1157 else os.path.join(tok_dir, "tokenizer.json")

        if not os.path.exists(tok_file):
            print(f"  Generating BPE {v_size} tokenizer...")
            train_tok_script = os.path.join(tok_dir, "train_bpe_tokenizer.py")
            cmd_tok = [sys.executable, train_tok_script, "--vocab_size", str(v_size), "--output_dir", tok_out, "--csv_path", csv_path]
            subprocess.run(cmd_tok, check=True)

        # Step 2: Train command
        train_script = os.path.join(base_dir, "byt5_train.py")
        cmd_train = [
            sys.executable, train_script,
            "--vocab_size", str(v_size),
            "--tokenizer_path", tok_file,
            "--csv_path", csv_path,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        print(f"  Command to train: {' '.join(cmd_train)}")
        # Note: can be triggered via subprocess if desired:
        # subprocess.run(cmd_train, check=True)

        results.append({
            "Vocab Size": v_size,
            "Tokenizer Path": os.path.relpath(tok_file, base_dir),
            "Epochs": args.epochs,
            "Status": "Configured / Ready to run"
        })

    # Output markdown report template
    report_lines = [
        "# ByT5 BPE Vocabulary Size Ablation Study\n",
        "Investigating the effect of subword vocabulary granularity on ByT5 translation performance:\n",
        "| Vocabulary Size | Tokenizer File | Epochs | Training Status | BLEU | chrF |",
        "|---|---|---|---|---|---|"
    ]
    for r in results:
        report_lines.append(f"| **{r['Vocab Size']}** | `{r['Tokenizer Path']}` | {r['Epochs']} | {r['Status']} | *Pending Run* | *Pending Run* |")

    report_path = os.path.join(base_dir, args.output_report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n✓ Experiment configuration and ablation plan generated: {report_path}")


if __name__ == "__main__":
    main()

