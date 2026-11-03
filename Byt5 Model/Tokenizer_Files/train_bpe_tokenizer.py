"""
train_bpe_tokenizer.py — Train BPE Tokenizers with Configurable Vocabulary Size
================================================================================
Trains a byte-pair encoding (BPE) tokenizer on the Gondi parallel corpus with
any requested vocabulary size (e.g., 1157, 2000, 4000, 8000, 16000, 32000).

Outputs:
  - tokenizer.json (HuggingFace Fast Tokenizer format)
  - tokenizer_config.json

Usage:
  python train_bpe_tokenizer.py --vocab_size 2000 --output_dir vocab_2000
  python train_bpe_tokenizer.py --vocab_size 4000 --output_dir vocab_4000
  python train_bpe_tokenizer.py --vocab_size 8000 --output_dir vocab_8000
"""

import os
import json
import argparse
import pandas as pd
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def train_bpe_tokenizer(csv_path: str, vocab_size: int, output_dir: str,
                        src_col: str = "Gondi", tgt_col1: str = "Hindi", tgt_col2: str = "English"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print(f"  🔧 Training BPE Tokenizer with Vocab Size: {vocab_size}")
    print(f"  📂 Dataset: {csv_path}")
    print(f"  💾 Output Directory: {output_dir}")
    print("=" * 65)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Corpus CSV not found at: {csv_path}")

    # Load corpus text
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts = []
    for col in [src_col, tgt_col1, tgt_col2]:
        if col in df.columns:
            cleaned = df[col].dropna().astype(str).str.strip()
            texts.extend(cleaned[cleaned != ""].tolist())

    print(f"  ✓ Loaded {len(texts):,} sentences for tokenizer training.")

    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["[UNK]", "[PAD]", "</s>", "[START]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if False else [],
        show_progress=True
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Save tokenizer.json
    out_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(out_file)

    # Save tokenizer_config.json
    config = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "eos_token": "</s>",
        "bos_token": "[START]",
        "vocab_size": tokenizer.get_vocab_size()
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"  ✓ BPE Tokenizer saved successfully!")
    print(f"    - File: {out_file}")
    print(f"    - Requested Vocab Size: {vocab_size}")
    print(f"    - Actual Learned Vocab Size: {actual_vocab_size}")
    print("=" * 65)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Train BPE Tokenizer with custom vocab size.")
    parser.add_argument("--vocab_size", type=int, default=2000,
                        help="Target vocabulary size (e.g. 1157, 2000, 4000, 8000, 16000).")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to corpus CSV.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory to save tokenizer files.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base_dir, "..", "..", "Datasets", "gondi_training_60k.csv")
    output_dir = args.output_dir or os.path.join(base_dir, f"vocab_{args.vocab_size}")

    train_bpe_tokenizer(csv_path, args.vocab_size, output_dir)


if __name__ == "__main__":
    main()

