import pandas as pd
from gondi_bpe_tokenizer import GondiTokenizer

def main():
    print("=" * 60)
    print("  🚀 Retraining BPE Tokenizer on 50k Dataset")
    print("=" * 60)
    
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load dataset
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "Datasets", "gondi_training_60k.csv"), encoding="utf-8-sig")
    
    # Gather text from Gondi + English (or Hindi)
    text_data = df["Gondi"].dropna().astype(str).tolist()
    
    if "English" in df.columns:
        print("Adding English text to tokenizer training...")
        text_data.extend(df["English"].dropna().astype(str).tolist())
    elif "Hindi" in df.columns:
        print("Adding Hindi text to tokenizer training...")
        text_data.extend(df["Hindi"].dropna().astype(str).tolist())

    print(f"Total sentences for tokenizer training: {len(text_data)}")

    # Initialize and train from scratch with larger vocab
    tokenizer = GondiTokenizer()
    VOCAB_SIZE = 5000
    
    print(f"Training tokenizer (Vocab Size: {VOCAB_SIZE}). This will take a moment...")
    tokenizer.train(text_data, vocab_size=VOCAB_SIZE)
    
    # Overwrite the old tokenizer
    tokenizer.save(os.path.join(BASE_DIR, "gondi_bpe_model.json"))
    print("✅ Tokenizer updated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
