"""
inference.py — Interactive Gondi -> Hindi/English Translation with GRU+Attention
================================================================================
"""

import os
import re
import sys
import torch
from gondi_bpe_tokenizer import GondiTokenizer
from gondi_gru_seq2seq import build_model

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH  = os.path.join(BASE_DIR, "gondi_bpe_model.json")
EMBED_DIM       = 256
HIDDEN_DIM      = 512
NUM_LAYERS      = 2
MAX_LEN         = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  🌿 Gondi GRU+Attention Translation Inference")
print("=" * 60)
print("  Choose Target Language:")
print("  1. Hindi")
print("  2. English")
choice = input("  Enter choice (1/2) [default: 1] ▶ ").strip() or "1"

target_lang = "English" if choice == "2" else "Hindi"
checkpoint_name = "best_gondi_gru_model_eng.pt" if choice == "2" else "best_gondi_gru_model.pt"
checkpoint_path = os.path.join(BASE_DIR, checkpoint_name)

if not os.path.exists(checkpoint_path):
    print(f"\n  ❌ Error: Checkpoint '{checkpoint_path}' not found!")
    print(f"  Please train the model first by running: python train.py")
    sys.exit(1)

tokenizer = GondiTokenizer.load(TOKENIZER_PATH)
PAD_IDX = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
SOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
EOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)
VOCAB_SIZE = tokenizer.vocab_size()

model = build_model(
    gondi_vocab_size=VOCAB_SIZE,
    target_vocab_size=VOCAB_SIZE,
    device=device,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=0.0,
    pad_idx=PAD_IDX
)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

print(f"  ✓ Device:    {device}")
print(f"  ✓ Direction: Gondi -> {target_lang}")
print(f"  ✓ Checkpoint: {checkpoint_name}")
print("=" * 60)

def translate(text: str) -> str:
    tokens = tokenizer.encode(text)
    if not tokens:
        return ""
    src_tensor = torch.tensor([[SOS_IDX] + tokens + [EOS_IDX]], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(tokens) + 2], dtype=torch.long, device=device)

    with torch.no_grad():
        preds, _ = model.translate(src_tensor, sos_idx=SOS_IDX, eos_idx=EOS_IDX, max_len=MAX_LEN, src_lengths=src_lengths)

    pred_ids = preds[0].tolist()
    words = []
    for tid in pred_ids:
        if tid == EOS_IDX:
            break
        if tid not in (PAD_IDX, SOS_IDX):
            w = tokenizer.id_to_token.get(tid, "")
            if w:
                words.append(w)
    return "".join(words).replace(" ", " ").strip()

print("\n  Type a Gondi sentence to translate (or 'exit' to quit):\n")
while True:
    try:
        user_input = input("  Gondi ▶ ").strip()
    except (KeyboardInterrupt, EOFError):
        break
    if not user_input or user_input.lower() in ("exit", "quit"):
        print("  Goodbye!")
        break
    result = translate(user_input)
    print(f"  ↳ {target_lang}: {result}\n")

