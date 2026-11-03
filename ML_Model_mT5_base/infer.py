import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR = "./byt5-gondi-final"   # path saved by train_byt5.py
MAX_INPUT_LEN  = 64
MAX_OUTPUT_LEN = 64
NUM_BEAMS      = 4                 # beam search width

# ── DIRECTION MENU ────────────────────────────────────────────────────────────
DIRECTIONS = {
    "1": ("English", "Gondi",   "Translate English to Gondi: {}"),
    "2": ("Gondi",   "English", "Translate Gondi to English: {}"),
    "3": ("Hindi",   "Gondi",   "Translate Hindi to Gondi: {}"),
    "4": ("Gondi",   "Hindi",   "Translate Gondi to Hindi: {}"),
    "5": ("English", "Hindi",   "Translate English to Hindi: {}"),
    "6": ("Hindi",   "English", "Translate Hindi to English: {}"),
}

def load_model():
    if not os.path.isdir(MODEL_DIR):
        print(f"\n[ERROR] Model directory '{MODEL_DIR}' not found.")
        print("        Please finish training first (train_byt5.py).\n")
        exit(1)

    print(f"\nLoading model from '{MODEL_DIR}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Running on : {gpu_name}")
    return tokenizer, model, device


def translate(text, prompt_template, tokenizer, model, device):
    prompt = prompt_template.format(text.strip())
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def print_banner():
    print("=" * 54)
    print("  ByT5 Gondi Translation Model -- Interactive Mode")
    print("=" * 54)


def pick_direction():
    print("\nChoose translation direction:")
    for key, (src, tgt, _) in DIRECTIONS.items():
        print(f"  [{key}]  {src}  -->  {tgt}")
    print("  [c]  Change direction")
    print("  [q]  Quit")
    print()

    while True:
        choice = input("Enter choice: ").strip().lower()
        if choice == "q":
            return None
        if choice in DIRECTIONS:
            src, tgt, template = DIRECTIONS[choice]
            print(f"\n>> Direction set: {src} --> {tgt}")
            print("   Type your text and press Enter. Type 'c' to change direction, 'q' to quit.\n")
            return choice
        print("   Invalid choice. Please enter 1-6, c, or q.")


def main():
    print_banner()
    tokenizer, model, device = load_model()

    direction_key = pick_direction()
    if direction_key is None:
        print("Goodbye!")
        return

    while True:
        src_lang = DIRECTIONS[direction_key][0]
        user_input = input(f"[{src_lang}] > ").strip()

        if not user_input:
            continue

        if user_input.lower() == "q":
            print("Goodbye!")
            break

        if user_input.lower() == "c":
            direction_key = pick_direction()
            if direction_key is None:
                print("Goodbye!")
                break
            continue

        _, tgt_lang, template = DIRECTIONS[direction_key]
        result = translate(user_input, template, tokenizer, model, device)
        print(f"[{tgt_lang}] > {result}\n")


if __name__ == "__main__":
    main()
