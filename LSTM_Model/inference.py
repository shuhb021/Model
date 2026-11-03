"""
inference.py — Interactive Gondi → Hindi/English Translation
=====================================================
Load the trained Seq2Seq model and translate Gondi input to target language.
Handles full sentences by translating word-by-word since the BPE
tokenizer's encode() preprocesses to single words.
"""

import re
import os
import torch
from gondi_bpe_tokenizer import GondiTokenizer
from gondi_seq2seq import build_model

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG (must match training)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH  = os.path.join(BASE_DIR, "gondi_bpe_model.json")
EMBED_DIM       = 256
HIDDEN_DIM      = 512
NUM_LAYERS      = 2
DROPOUT         = 0.0   # no dropout at inference
MAX_LEN         = 150   # max output tokens

# ─────────────────────────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  🌿 Gondi Translation Inference")
print("=" * 60)
print("  Choose Target Language:")
print("  1. Hindi")
print("  2. English")
choice = input("  Enter choice (1/2) ▶ ").strip()

target_lang = "English" if choice == "2" else "Hindi"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_gondi_model.pt") if choice == "1" else os.path.join(BASE_DIR, "best_gondi_model_eng.pt")

print("\n  Loading tokenizer & model …")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"\n  ❌ Error: Checkpoint '{CHECKPOINT_PATH}' not found!")
    if choice == "2":
        print("  💡 Tip: You need to train an English model first.")
        print("  Change 'tgt_col=\"English\"' and CHECKPOINT_PATH=\"best_gondi_model_eng.pt\" in train.py and run it.")
    exit(1)

tokenizer = GondiTokenizer.load(TOKENIZER_PATH)

PAD_IDX = tokenizer.token_to_id[GondiTokenizer.PAD_TOKEN]
SOS_IDX = tokenizer.token_to_id[GondiTokenizer.START_TOKEN]
EOS_IDX = tokenizer.token_to_id[GondiTokenizer.END_TOKEN]
UNK_IDX = tokenizer.token_to_id[GondiTokenizer.UNK_TOKEN]
VOCAB_SIZE = tokenizer.vocab_size()

# Build model & load weights
model = build_model(
    gondi_vocab_size  = VOCAB_SIZE,
    target_vocab_size = VOCAB_SIZE,
    device            = device,
    embed_dim         = EMBED_DIM,
    hidden_dim        = HIDDEN_DIM,
    num_layers        = NUM_LAYERS,
    dropout           = DROPOUT,
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"  Device    : {device}")
print(f"  Language  : Gondi → {target_lang}")
print(f"  Loaded    : {CHECKPOINT_PATH} (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODE FULL SENTENCE (bypass single-word preprocess)
# ─────────────────────────────────────────────────────────────────────────────
def encode_sentence(text: str) -> list:
    """
    Encode a full Gondi sentence by processing each word through the
    tokenizer's internal BPE pipeline, bypassing preprocess_gondi_text()
    which truncates to a single word.
    """
    # Light cleanup — keep all words, remove punctuation
    text = text.strip().lstrip("\ufeff\u200b")
    text = re.sub(r"[।,.!?;:\"'()\[\]{}]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    all_ids = []
    unk_id = tokenizer.token_to_id.get(GondiTokenizer.UNK_TOKEN, UNK_IDX)

    for word in words:
        # Use the internal _tokenize_word for each word
        # (this applies BPE merges without the first-word truncation)
        tokens = tokenizer._tokenize_word(word.lower())
        for tok in tokens:
            all_ids.append(tokenizer.token_to_id.get(tok, unk_id))

    return all_ids


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def translate(text: str) -> str:
    """Translate a Gondi sentence to target language."""
    # Encode full sentence
    src_ids = encode_sentence(text)
    if not src_ids:
        return "[Empty input — no tokens found]"

    # Add SOS/EOS
    src_ids = [SOS_IDX] + src_ids + [EOS_IDX]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long)

    # Translate using model
    with torch.no_grad():
        token_ids, attn_weights = model.translate(
            src_tensor,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            max_len=MAX_LEN,
            src_lengths=src_lengths,
        )

    # Decode output — stop at EOS
    output_ids = token_ids[0].tolist()
    if EOS_IDX in output_ids:
        output_ids = output_ids[:output_ids.index(EOS_IDX)]

    # Remove PAD and special tokens
    output_ids = [tid for tid in output_ids
                  if tid not in (PAD_IDX, SOS_IDX, EOS_IDX)]

    result = tokenizer.decode(output_ids)
    return result if result.strip() else "[No translation generated]"


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n🌿 Gondi → {target_lang} Translator")
    print("   Type a Gondi sentence and press Enter.")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("  Gondi  ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye! 👋")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("  Bye! 👋")
            break

        output_text = translate(text)
        print(f"  {target_lang:<7}◀ {output_text}\n")


if __name__ == "__main__":
    main()
