"""
byt5_train.py — ByT5 Translation Model Training with Configurable BPE Vocab Size
================================================================================
Supports training ByT5 on the Gondi parallel corpus with:
  - Original BPE 1157 token vocabulary
  - Custom / ablated vocabulary sizes (e.g. 2000, 4000, 8000, etc.)

Usage:
  # Train with default BPE 1157 tokens:
  python byt5_train.py --vocab_size 1157 --epochs 5

  # Train with custom vocabulary size (e.g., 2000, 4000, 8000):
  python byt5_train.py --vocab_size 2000 --epochs 5
  python byt5_train.py --vocab_size 4000 --epochs 5
"""

import os
import sys
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ByT5 model with configurable BPE vocab size.")
    parser.add_argument("--vocab_size", type=int, default=1157,
                        help="BPE vocabulary size (e.g. 1157, 2000, 4000, 8000).")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Explicit path to tokenizer.json. If None, resolves automatically.")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to training CSV (default: ../../Datasets/gondi_training_60k.csv).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length.")
    parser.add_argument("--d_model", type=int, default=384, help="T5 hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of encoder layers.")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of decoder layers.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for checkpoints.")
    parser.add_argument("--save_path", type=str, default=None, help="Directory for final saved model.")
    return parser.parse_args()


def resolve_tokenizer_path(base_dir: str, vocab_size: int, explicit_path: str = None) -> str:
    """Finds or prepares tokenizer.json matching requested vocab size."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    # Candidate 1: Specific vocab directory under Tokenizer_Files/
    cand1 = os.path.join(base_dir, "..", "Tokenizer_Files", f"vocab_{vocab_size}", "tokenizer.json")
    if os.path.exists(cand1):
        return cand1

    # Candidate 2: Default tokenizer.json in Tokenizer_Files
    cand2 = os.path.join(base_dir, "..", "Tokenizer_Files", "tokenizer.json")
    if vocab_size == 1157 and os.path.exists(cand2):
        return cand2

    # Candidate 3: Same directory as script
    cand3 = os.path.join(base_dir, "tokenizer.json")
    if os.path.exists(cand3):
        return cand3

    # If not found and vocab_size is different, train on the fly or fail with clear instruction
    print(f"  ⚠️ Tokenizer for vocab_size={vocab_size} not found at {cand1}.")
    print(f"  💡 Automatically training BPE tokenizer with vocab_size={vocab_size}...")

    try:
        from Tokenizer_Files.train_bpe_tokenizer import train_bpe_tokenizer
    except ImportError:
        tok_dir = os.path.join(base_dir, "..", "Tokenizer_Files")
        sys.path.append(tok_dir)
        from train_bpe_tokenizer import train_bpe_tokenizer

    csv_path = os.path.join(base_dir, "..", "..", "Datasets", "gondi_training_60k.csv")
    out_dir = os.path.join(base_dir, "..", "Tokenizer_Files", f"vocab_{vocab_size}")
    trained_path = train_bpe_tokenizer(csv_path, vocab_size, out_dir)
    return trained_path


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = args.csv_path or os.path.join(base_dir, "..", "..", "Datasets", "gondi_training_60k.csv")
    if not os.path.exists(csv_path):
        # Fallback to single parent
        csv_path = os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")

    tokenizer_path = resolve_tokenizer_path(base_dir, args.vocab_size, args.tokenizer_path)

    output_dir = args.output_dir or os.path.join(base_dir, f"byt5_gondi_results_v{args.vocab_size}")
    save_path = args.save_path or os.path.join(base_dir, f"byt5_gondi_model_v{args.vocab_size}")

    print("=" * 65)
    print(f" 🚀 ByT5 Training — BPE Vocab Size: {args.vocab_size}")
    print(f" 📂 Loading BPE Tokenizer from: {tokenizer_path}")
    print("=" * 65)

    # Load BPE tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="</s>",
        bos_token="[START]"
    )

    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded. Exact vocabulary size: {vocab_size}")

    print("\n" + "=" * 65)
    print(f" Loading data from {csv_path}...")
    print("=" * 65)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(subset=["Gondi", "Hindi", "English"]).reset_index(drop=True)

    # Multilingual translation pairs
    pairs = [
        ("Gondi", "English", "translate Gondi to English: "),
        ("Gondi", "Hindi", "translate Gondi to Hindi: "),
        ("Hindi", "Gondi", "translate Hindi to Gondi: "),
        ("English", "Gondi", "translate English to Gondi: ")
    ]

    all_data = []
    for src, tgt, prefix in pairs:
        temp_df = pd.DataFrame({
            "source": prefix + df[src].astype(str) + " </s>",
            "target": df[tgt].astype(str) + " </s>"
        })
        all_data.append(temp_df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total sentences for multi-language training: {len(combined_df):,}")

    dataset = Dataset.from_pandas(combined_df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        model_inputs = tokenizer(inputs, max_length=args.max_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=args.max_length, padding="max_length", truncation=True)

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f"\nTokenizing dataset with BPE vocabulary ({vocab_size} tokens)...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])

    # Configure ByT5 architecture
    print(f"\nInitializing ByT5 architecture with BPE vocab_size={vocab_size}...")
    config = T5Config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=1024,
        d_kv=64,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=6,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    model = T5ForConditionalGeneration(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\n" + "=" * 65)
    print(f" Starting ByT5 (Vocab: {vocab_size}) training loop...")
    print("=" * 65)

    resume_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            resume_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"  ↳ Resuming from checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print(f"\nSaving final model (vocab {vocab_size}) to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Done! ByT5 model and BPE tokenizer saved to {save_path}.")


if __name__ == "__main__":
    main()
