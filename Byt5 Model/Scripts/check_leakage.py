import pandas as pd
import os

train_path = r"d:\New models Lokesh sir\Datasets\gondi_training_60k.csv"
test_path = r"d:\New models Lokesh sir\Datasets\gondi_test_40k_fixed.csv"

def check_leakage():
    print(f"[*] Loading training set: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"[*] Loading test set: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Extract Gondi sentences
    train_sentences = set(train_df['Gondi'].astype(str).tolist())
    test_sentences = test_df['Gondi'].astype(str).tolist()
    
    total_test = len(test_sentences)
    leaked_count = 0
    leaked_examples = []
    
    print("[*] Checking for overlaps...")
    for s in test_sentences:
        if s in train_sentences:
            leaked_count += 1
            if len(leaked_examples) < 5:
                leaked_examples.append(s)
                
    leak_percent = (leaked_count / total_test) * 100
    
    print("\n" + "="*50)
    print("           DATA LEAKAGE REPORT")
    print("="*50)
    print(f"Total Training Sentences: {len(train_sentences)}")
    print(f"Total Test Sentences:     {total_test}")
    print(f"Number of Leaked Rows:    {leaked_count}")
    print(f"Leakage Percentage:       {leak_percent:.2f}%")
    print("="*50)
    
    if leaked_examples:
        print("\n[*] Examples of leaked sentences (found in both files):")
        for i, ex in enumerate(leaked_examples, 1):
            print(f"{i}. {ex}")
    
    if leak_percent > 50:
        print("\n[WARNING] HIGH LEAKAGE DETECTED!")
        print("Your model's high BLEU score is likely because it has already seen most of the test data during training.")
    else:
        print("\n[INFO] Leakage is within acceptable or low range if this is a random split.")

if __name__ == "__main__":
    check_leakage()
