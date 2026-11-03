"""
inference.py — Interactive Translation Inference for Conventional Baselines
============================================================================
Translates Gondi sentences to Hindi or English using trained baseline models:
  - BiLSTM + Bahdanau Attention
  - GRU + Bahdanau Attention
  - Standard Transformer-Small
  - Standard Transformer-Base

Usage:
  # Interactive mode:
  python inference.py --model bilstm --checkpoint checkpoints/best_bilstm_hindi.pt

  # Single sentence mode:
  python inference.py --model gru --checkpoint checkpoints/best_gru_hindi.pt --text "मना नाव राहुल"
"""

import os
import sys
import time
import argparse
import torch

from gondi_bpe_tokenizer import GondiTokenizer
from models import (
    build_bilstm_model,
    build_gru_model,
    build_transformer_small,
    build_transformer_base,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Gondi Seq2Seq Baselines.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["bilstm", "gru", "transformer_small", "transformer_base"],
                        help="Baseline model type.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to BPE tokenizer JSON.")
    parser.add_argument("--tgt_lang", type=str, default="Hindi", choices=["Hindi", "English"])
    parser.add_argument("--text", type=str, default=None, help="Single Gondi sentence to translate.")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum generated tokens.")
    return parser.parse_args()


def load_model_for_inference(model_type: str, checkpoint_path: str, vocab_size: int, device: torch.device):
    pad_idx = 0
    if model_type == "bilstm":
        model = build_bilstm_model(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif model_type == "gru":
        model = build_gru_model(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif model_type == "transformer_small":
        model = build_transformer_small(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    elif model_type == "transformer_base":
        model = build_transformer_base(vocab_size=vocab_size, device=device, pad_idx=pad_idx)
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


def translate_sentence(text: str, model, tokenizer: GondiTokenizer, is_transformer: bool,
                       device: torch.device, max_len: int = 100) -> str:
    pad_idx = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
    sos_idx = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
    eos_idx = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)

    clean_text = text.strip()
    src_ids = tokenizer.encode(clean_text)
    if not src_ids:
        return ""

    src_tensor = torch.tensor([[sos_idx] + src_ids + [eos_idx]], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(src_ids) + 2], dtype=torch.long, device=device)

    with torch.no_grad():
        if is_transformer:
            pred_tokens, _ = model.translate(src_tensor, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)
        else:
            pred_tokens, _ = model.translate(src_tensor, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len, src_lengths=src_lengths)

    gen_ids = pred_tokens[0].tolist()
    tokens = []
    for tid in gen_ids:
        if tid == eos_idx:
            break
        if tid in (pad_idx, sos_idx):
            continue
        tok = tokenizer.id_to_token.get(tid, "")
        if tok:
            tokens.append(tok)

    result = "".join(tokens).replace(" ", " ").strip()
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = args.tokenizer_path or os.path.join(base_dir, "gondi_bpe_model.json")

    print("=" * 65)
    print("  🌿 Gondi Baseline Translation Inference")
    print(f"  🖥️  Device: {device}")
    print("=" * 65)

    tokenizer = GondiTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size()

    # Model selection if not provided
    model_type = args.model
    if not model_type:
        print("\n  Select baseline architecture:")
        print("    1. BiLSTM + Bahdanau Attention")
        print("    2. GRU + Bahdanau Attention")
        print("    3. Standard Transformer-Small")
        print("    4. Standard Transformer-Base")
        choice = input("  Choice (1-4) [default: 1] ▶ ").strip() or "1"
        mapping = {"1": "bilstm", "2": "gru", "3": "transformer_small", "4": "transformer_base"}
        model_type = mapping.get(choice, "bilstm")

    # Checkpoint path
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        default_ckpt = os.path.join(base_dir, "checkpoints", f"best_{model_type}_{args.tgt_lang.lower()}.pt")
        if not os.path.exists(default_ckpt):
            # Also check parent LSTM_Model or root if available
            alt = os.path.join(base_dir, "..", "LSTM_Model", "best_gondi_model.pt")
            default_ckpt = alt if os.path.exists(alt) and model_type == "bilstm" else default_ckpt

        user_ckpt = input(f"  Enter checkpoint path [default: {default_ckpt}] ▶ ").strip()
        checkpoint_path = user_ckpt or default_ckpt

    if not os.path.exists(checkpoint_path):
        print(f"\n  ❌ Error: Checkpoint not found at: {checkpoint_path}")
        print("  💡 Tip: Train the model first with train.py, for example:")
        print(f"     python train.py --model {model_type} --tgt_lang {args.tgt_lang}")
        sys.exit(1)

    is_transformer = "transformer" in model_type
    model, ckpt_meta = load_model_for_inference(model_type, checkpoint_path, vocab_size, device)
    params = sum(p.numel() for p in model.parameters())

    print(f"\n  ✓ Loaded: {checkpoint_path}")
    print(f"  ✓ Architecture: {model_type.upper()} ({params:,} parameters)")
    print(f"  ✓ Direction: Gondi -> {args.tgt_lang}")
    print("=" * 65)

    if args.text:
        t0 = time.time()
        translation = translate_sentence(args.text, model, tokenizer, is_transformer, device, args.max_len)
        ms = (time.time() - t0) * 1000.0
        print(f"\n  Gondi (Input):  {args.text}")
        print(f"  {args.tgt_lang} (Output): {translation}")
        print(f"  Latency:        {ms:.1f} ms\n")
        return

    print("\n  Type a Gondi sentence to translate (or 'exit' / 'quit' to exit):\n")
    while True:
        try:
            line = input("  Gondi ▶ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if not line or line.lower() in ("exit", "quit", "q"):
            print("  Goodbye!")
            break

        t0 = time.time()
        result = translate_sentence(line, model, tokenizer, is_transformer, device, args.max_len)
        ms = (time.time() - t0) * 1000.0

        print(f"  ↳ {args.tgt_lang} : {result}  ({ms:.1f} ms)\n")


if __name__ == "__main__":
    main()

