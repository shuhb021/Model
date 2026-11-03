import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from gondi_bpe_tokenizer import GondiTokenizer
from gondi_seq2seq import build_model
import random

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load tokenizer
    tokenizer = GondiTokenizer.load(os.path.join(BASE_DIR, "gondi_bpe_model.json"))
    VOCAB_SIZE = tokenizer.vocab_size()
    PAD_IDX = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
    SOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
    EOS_IDX = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)

    # Build model
    model = build_model(
        gondi_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        device=device,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.0
    )

    checkpoint_path = os.path.join(BASE_DIR, "best_gondi_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "Datasets", "gondi_training_60k.csv"))
    df = df.dropna(subset=["Gondi", "Hindi"]) # Assuming Gondi -> Hindi
    
    # Sample 1000 rows
    df_sample = df.sample(n=1000, random_state=42)
    
    true_tokens_all = []
    pred_tokens_all = []
    
    print("Generating predictions for 1000 samples...")
    with torch.no_grad():
        for idx, row in enumerate(df_sample.itertuples()):
            src_text = str(row.Gondi)
            tgt_text = str(row.Hindi)
            
            src_ids = tokenizer.encode(src_text)
            tgt_ids = tokenizer.encode(tgt_text)
            
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                continue
                
            src_tensor = torch.tensor([[SOS_IDX] + src_ids + [EOS_IDX]], dtype=torch.long, device=device)
            src_lengths = torch.tensor([len(src_ids) + 2], dtype=torch.long)
            
            try:
                pred_ids_batch, _ = model.translate(
                    src_tensor,
                    sos_idx=SOS_IDX,
                    eos_idx=EOS_IDX,
                    max_len=150,
                    src_lengths=src_lengths
                )
            except Exception as e:
                print(f"Error during translation: {e}")
                continue
                
            pred_ids = pred_ids_batch[0].tolist()
            if EOS_IDX in pred_ids:
                pred_ids = pred_ids[:pred_ids.index(EOS_IDX)]
            pred_ids = [tid for tid in pred_ids if tid not in (PAD_IDX, SOS_IDX)]
            
            # Since seq2seq outputs can have different lengths, we align them up to the minimum length
            # or pad predictions with a special token. Simple way: zip true and pred.
            min_len = min(len(tgt_ids), len(pred_ids))
            
            true_tokens_all.extend(tgt_ids[:min_len])
            pred_tokens_all.extend(pred_ids[:min_len])
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} / 1000")

    if not true_tokens_all:
        print("No tokens collected!")
        return

    # To avoid a massive confusion matrix, let's pick the top 20 most frequent tokens in the true set
    from collections import Counter
    true_counts = Counter(true_tokens_all)
    top_n = 20
    top_tokens = [tok for tok, count in true_counts.most_common(top_n)]
    
    other_token_id = -1
    
    def map_token(tid):
        return tid if tid in top_tokens else other_token_id

    mapped_true = [map_token(tid) for tid in true_tokens_all]
    mapped_pred = [map_token(tid) for tid in pred_tokens_all]
    
    labels = top_tokens + [other_token_id]
    label_names = []
    for tid in labels:
        if tid == other_token_id:
            label_names.append("OTHER")
        else:
            token_str = tokenizer.id_to_token.get(tid, f"UNK_{tid}")
            label_names.append(token_str)
            
    cm = confusion_matrix(mapped_true, mapped_pred, labels=labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Tokens')
    plt.ylabel('True Tokens')
    plt.title('Confusion Matrix for Top 20 Tokens (1000 samples)')
    plt.tight_layout()
    out_img = os.path.join(BASE_DIR, 'confusion_matrix_1000.png')
    plt.savefig(out_img, dpi=300)
    print(f"Saved confusion matrix as {out_img}")

if __name__ == "__main__":
    main()
