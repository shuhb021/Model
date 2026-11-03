"""
run_all_experiments.py — Master Experiment Runner for Reviewer Requirements
=============================================================================
Addresses all 3 tasks:
  P1: Run ByT5 with different BPE vocabulary sizes (e.g., 1157, 2000, 4000, 8000).
  P2: Run Vanilla Transformer with the same dataset and BPE tokenizer.
  P3: Run BiLSTM + Attention and GRU + Attention with the same corpus.

Usage:
  # Inspect experiment configurations and commands without running:
  python run_all_experiments.py --dry_run

  # Run P1 only (ByT5 vocabulary ablation):
  python run_all_experiments.py --run_p1 --vocabs 1157 2000 4000 --epochs 5

  # Run P2 only (Vanilla Transformer):
  python run_all_experiments.py --run_p2 --epochs 30

  # Run P3 only (BiLSTM+Attention & GRU+Attention):
  python run_all_experiments.py --run_p3 --epochs 30

  # Run all three experiment sets:
  python run_all_experiments.py --all --epochs 30
"""

import os
import sys
import argparse
import subprocess
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Master Experiment Runner for Gondi Baselines & ByT5 Ablations.")
    parser.add_argument("--run_p1", action="store_true", help="P1: Run ByT5 with different BPE vocab sizes.")
    parser.add_argument("--run_p2", action="store_true", help="P2: Run Vanilla Transformer on same corpus.")
    parser.add_argument("--run_p3", action="store_true", help="P3: Run BiLSTM+attn and GRU+attn on same corpus.")
    parser.add_argument("--all", action="store_true", help="Run all experiments (P1, P2, and P3).")
    parser.add_argument("--vocabs", nargs="+", type=int, default=[1157, 2000, 4000, 8000],
                        help="Vocabulary sizes for P1 ByT5 ablation.")
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"])
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print planned execution commands without running.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if any flag provided
    run_all = args.all or not (args.run_p1 or args.run_p2 or args.run_p3)
    do_p1 = run_all or args.run_p1
    do_p2 = run_all or args.run_p2
    do_p3 = run_all or args.run_p3

    print("=" * 80)
    print("  🔬 MASTER EXPERIMENT RUNNER — REVIEWER REQUIREMENTS (P1, P2, P3)")
    print("=" * 80)
    print(f"  Target Language: Gondi -> {args.tgt_lang}")
    print(f"  Dry Run Mode:    {args.dry_run} (Code configured without execution)")
    print("=" * 80)

    commands_to_run = []

    # ─────────────────────────────────────────────────────────────────────────
    # P1: ByT5 Vocabulary Size Ablation
    # ─────────────────────────────────────────────────────────────────────────
    if do_p1:
        print("\n[P1] ByT5 with Configurable BPE Vocabulary Sizes:")
        tok_dir = os.path.join(base_dir, "Byt5 Model", "Tokenizer_Files")
        byt5_script = os.path.join(base_dir, "Byt5 Model", "Scripts", "byt5_train.py")
        csv_path = os.path.join(base_dir, "Datasets", "gondi_training_60k.csv")

        for v in args.vocabs:
            tok_path = os.path.join(tok_dir, "tokenizer.json") if v == 1157 else os.path.join(tok_dir, f"vocab_{v}", "tokenizer.json")
            cmd = [
                sys.executable, byt5_script,
                "--vocab_size", str(v),
                "--tokenizer_path", tok_path,
                "--csv_path", csv_path,
                "--epochs", str(min(args.epochs, 5)),
                "--batch_size", str(min(args.batch_size, 16))
            ]
            commands_to_run.append(("P1: ByT5 Vocab " + str(v), cmd))
            print(f"  • Vocab {v:>5}: python \"{os.path.relpath(byt5_script, base_dir)}\" --vocab_size {v}")

    # ─────────────────────────────────────────────────────────────────────────
    # P2: Vanilla Transformer on Same Data & BPE Tokenizer
    # ─────────────────────────────────────────────────────────────────────────
    if do_p2:
        print("\n[P2] Vanilla Transformer with Same Corpus & BPE Tokenizer:")
        trans_script = os.path.join(base_dir, "Transformer_Model", "vanilla_transformer.py")
        tok_path = os.path.join(base_dir, "Byt5 Model", "Tokenizer_Files", "tokenizer.json")
        cmd_p2 = [
            sys.executable, trans_script,
            "--train",
            "--tokenizer_path", tok_path,
            "--tgt_lang", args.tgt_lang,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        commands_to_run.append(("P2: Vanilla Transformer (1157 BPE)", cmd_p2))
        print(f"  • Vanilla Transformer: python \"{os.path.relpath(trans_script, base_dir)}\" --train --tokenizer_path \"{os.path.relpath(tok_path, base_dir)}\"")

    # ─────────────────────────────────────────────────────────────────────────
    # P3: BiLSTM + Attention and GRU + Attention with Same Corpus
    # ─────────────────────────────────────────────────────────────────────────
    if do_p3:
        print("\n[P3] Conventional Recurrent Baselines with Same Corpus:")
        conv_train = os.path.join(base_dir, "Conventional_Baselines", "train.py")
        tok_1157 = os.path.join(base_dir, "Byt5 Model", "Tokenizer_Files", "tokenizer.json")

        cmd_bilstm = [
            sys.executable, conv_train,
            "--model", "bilstm",
            "--tokenizer_path", tok_1157,
            "--tgt_lang", args.tgt_lang,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        cmd_gru = [
            sys.executable, conv_train,
            "--model", "gru",
            "--tokenizer_path", tok_1157,
            "--tgt_lang", args.tgt_lang,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        commands_to_run.append(("P3: BiLSTM + Attention", cmd_bilstm))
        commands_to_run.append(("P3: GRU + Attention", cmd_gru))
        print(f"  • BiLSTM + Attention: python \"{os.path.relpath(conv_train, base_dir)}\" --model bilstm --tgt_lang {args.tgt_lang}")
        print(f"  • GRU + Attention:    python \"{os.path.relpath(conv_train, base_dir)}\" --model gru --tgt_lang {args.tgt_lang}")

    print("\n" + "=" * 80)
    print("  📋 EXPERIMENT PLAN & CONFIGURATION MATRIX")
    print("=" * 80)
    matrix = [
        {"Experiment": "P1 (ByT5)", "Model": "ByT5-Small (Custom BPE)", "Vocabulary": "1157, 2000, 4000, 8000", "Corpus": "Datasets/gondi_training_60k.csv", "Script": "Byt5 Model/Scripts/byt5_train.py"},
        {"Experiment": "P2 (Vanilla)", "Model": "Vanilla Transformer (Base/Small)", "Vocabulary": "1157 BPE / 5000 BPE", "Corpus": "Datasets/gondi_training_60k.csv", "Script": "Transformer_Model/vanilla_transformer.py"},
        {"Experiment": "P3 (BiLSTM)", "Model": "BiLSTM + Bahdanau Attention", "Vocabulary": "1157 BPE / 5000 BPE", "Corpus": "Datasets/gondi_training_60k.csv", "Script": "Conventional_Baselines/train.py --model bilstm"},
        {"Experiment": "P3 (GRU)", "Model": "GRU + Bahdanau Attention", "Vocabulary": "1157 BPE / 5000 BPE", "Corpus": "Datasets/gondi_training_60k.csv", "Script": "Conventional_Baselines/train.py --model gru"},
    ]
    df = pd.DataFrame(matrix)
    print(df.to_string(index=False))
    print("=" * 80)

    if args.dry_run:
        print("\n  🔒 Dry run flag active: No training commands executed.")
        print("  All code files, tokenizers, configs, and pipelines are updated and ready for execution.")
        return

    # Execute if dry_run is False and user specifically requested execution
    print("\nExecuting experiments...")
    for label, cmd in commands_to_run:
        print(f"\n▶ Running: {label}...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

