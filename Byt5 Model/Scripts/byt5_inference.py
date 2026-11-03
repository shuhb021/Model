import os
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

# ─────────────────────────────────────────────────────────────────────────────
# All supported translation directions (trained pairs)
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATION_MODES = {
    "1": {"prefix": "translate Gondi to English: ", "src": "Gondi",   "tgt": "English"},
    "2": {"prefix": "translate Gondi to Hindi: ",   "src": "Gondi",   "tgt": "Hindi"},
    "3": {"prefix": "translate Hindi to Gondi: ",   "src": "Hindi",   "tgt": "Gondi"},
    "4": {"prefix": "translate English to Gondi: ", "src": "English", "tgt": "Gondi"},
}

def translate(text, model, tokenizer, device, prefix):
    """Run inference for a single sentence."""
    input_text = prefix + text.strip() + " </s>"
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=128, truncation=True, padding=True
    ).to(device)

    with torch.no_grad():
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_banner():
    print("\n" + "═" * 56)
    print("       ByT5 Gondi Multi-Language Translator")
    print("═" * 56)
    print("  Supports: Gondi ↔ Hindi ↔ English")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'menu' to re-show language options.")
    print("═" * 56)


def print_menu():
    print("\n┌─ Select Translation Direction ─────────────────────┐")
    for key, mode in TRANSLATION_MODES.items():
        print(f"│  [{key}]  {mode['src']:8s} → {mode['tgt']}")
    print("└────────────────────────────────────────────────────┘")


def main():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "byt5_gondi_model")
    tok_path   = os.path.join(base_dir, "tokenizer.json")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run byt5_train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Using device: {device}")
    print("[*] Loading model and tokenizer — please wait...")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="</s>",
        bos_token="[START]"
    )
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    print("[*] Model loaded successfully!")

    print_banner()

    # ── Interactive loop ──────────────────────────────────────────────────────
    current_mode = None

    while True:
        # If no mode selected yet, show menu and ask
        if current_mode is None:
            print_menu()
            choice = input("\n  Enter choice (1-4): ").strip()

            if choice.lower() in ("quit", "exit"):
                print("\n[*] Goodbye!\n")
                break
            if choice not in TRANSLATION_MODES:
                print("  ⚠ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue

            current_mode = choice

        mode = TRANSLATION_MODES[current_mode]
        src_lang = mode["src"]
        tgt_lang = mode["tgt"]

        print(f"\n  Mode: {src_lang} → {tgt_lang}")
        print(f"  (type 'menu' to switch, 'quit' to exit)\n")

        while True:
            try:
                user_input = input(f"  [{src_lang}] ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[*] Interrupted. Goodbye!\n")
                return

            if user_input.lower() in ("quit", "exit"):
                print("\n[*] Goodbye!\n")
                return

            if user_input.lower() == "menu":
                current_mode = None
                break  # break inner loop → go back to menu

            if not user_input:
                print("  ⚠ Please enter some text.")
                continue

            # Translate
            try:
                result = translate(user_input, model, tokenizer, device, mode["prefix"])
                print(f"  [{tgt_lang}] ✓ {result}\n")
            except Exception as e:
                print(f"  [ERROR] Translation failed: {e}\n")


if __name__ == "__main__":
    main()
