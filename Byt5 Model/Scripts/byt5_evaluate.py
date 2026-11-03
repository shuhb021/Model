"""
Multilingual ByT5-Inspired T5 Model Evaluation Pipeline
=========================================================
Script Location: Byt5 Model/byt5_evaluate.py

This script performs a comprehensive, production-ready evaluation of the 
custom ByT5-inspired translation model on exactly 500 test samples.

Metrics Computed:
1. BLEU: Classic n-gram overlap metric. 
   Limitation: Fails to capture semantic meaning; penalizes valid paraphrases.
   Why BLEU alone is insufficient: It counts exact n-gram matches. A perfect 
   paraphrase that conveys identical meaning but uses different words gets a 
   score of 0. For low-resource languages like Gondi, this is especially 
   misleading since many valid translations exist.

2. chrF: Character n-gram F-score. Better for morphologically rich languages 
   like Gondi where word boundaries are complex.

3. METEOR: Uses stemming and synonyms for more flexible matching. Addresses 
   some of BLEU's limitations.

4. BERTScore: Uses contextual embeddings to measure semantic similarity.
   Advantage: Semantically stronger than BLEU. A synonym scores nearly as high
   as the exact word. Best for evaluating meaning preservation.

5. COMET: State-of-the-art neural metric trained on human judgments.
   Advantage: Correlates best with actual human evaluation. Considers source 
   sentence quality, not just surface-level overlap.

6. Token-Level P/R/F1/F2: Measures exact token overlap at the word level.

Confusion Matrix in Translation:
========================================================
Unlike classification where there are fixed classes, translation has an open 
vocabulary. To adapt the confusion matrix concept:
- We extract the top 20 most frequent tokens in the reference translations.
- All other tokens are mapped to "OTHER".
- We compare the token-by-token alignment of reference vs prediction.
- This reveals which specific tokens the model consistently gets right (TP) or 
  confuses with other tokens (FP, FN).
========================================================
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

# Evaluation Libraries
import sacrebleu
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint
import nltk
from nltk.translate.meteor_score import meteor_score

# Ensure NLTK wordnet is downloaded for METEOR
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
# New structure paths
MODEL_DIR      = os.path.join(BASE_DIR, "..", "Model_Weights")
TOKENIZER_DIR  = os.path.join(BASE_DIR, "..", "Tokenizer_Files")
CSV_FILE       = os.path.join(BASE_DIR, "..", "..", "Datasets", "gondi_test_40k_fixed.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "..", "results Byt5")

SRC_COL  = "Gondi"
TGT_COL  = "English"
PREFIX   = "translate Gondi to English: "
N_SAMPLES = 1000 # Increased for the 40k test set

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_dir, tokenizer_dir, device):
    """Load the trained ByT5 BPE model and tokenizer."""
    print(f"[*] Loading tokenizer from {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="</s>",
        bos_token="[START]"
    )

    print(f"[*] Loading model from {model_dir}")
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_predictions(model, tokenizer, df, device, batch_size=16):
    """
    Runs batched inference using model.generate() with beam search.
    Uses mixed precision (FP16) automatically if CUDA is available.
    """
    sources, references, predictions = [], [], []

    print(f"[*] Running inference on {len(df)} samples with batch_size={batch_size}...")

    for i in tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch = df.iloc[i:i + batch_size]
        src_texts = batch[SRC_COL].astype(str).tolist()
        tgt_texts = batch[TGT_COL].astype(str).tolist()

        input_texts = [PREFIX + t for t in src_texts]

        inputs = tokenizer(
            input_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128
        ).to(device)

        with torch.no_grad():
            # Mixed precision inference
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        sources.extend(src_texts)
        references.extend(tgt_texts)
        predictions.extend(preds)

        # GPU memory cleanup
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sources, references, predictions

# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def compute_bleu(references, predictions):
    """
    SacreBLEU: Standardized BLEU implementation.
    Returns score in [0, 100].
    """
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score

def compute_chrf(references, predictions):
    """
    SacreBLEU chrF: Character n-gram F-score.
    More robust than BLEU for low-resource & morphologically rich languages.
    Returns score in [0, 100].
    """
    result = sacrebleu.corpus_chrf(predictions, [references])
    return result.score

def compute_meteor(references, predictions):
    """
    NLTK METEOR: Uses unigram alignment, stemming, synonymy.
    Returns score in [0, 100].
    """
    scores = [
        meteor_score([ref.split()], pred.split())
        for ref, pred in zip(references, predictions)
    ]
    return (sum(scores) / len(scores)) * 100

def compute_bert_score(references, predictions):
    """
    BERTScore: Uses BERT contextual embeddings to compare translations.
    Returns F1 score. Range: generally [0.8, 1.0] for good translations.
    Semantically stronger than BLEU: paraphrases and synonyms score well.
    """
    P, R, F1 = bert_score(predictions, references, lang="en", device="cpu", verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def compute_comet(sources, references, predictions, comet_model, device):
    """
    COMET: Neural MT metric trained on human judgments.
    Most correlated with human evaluation. Considers source quality.
    Returns system-level score.
    """
    comet_data = [
        {"src": src, "mt": pred, "ref": ref}
        for src, pred, ref in zip(sources, predictions, references)
    ]
    # Force CPU (gpus=0) for COMET as well
    result = comet_model.predict(comet_data, batch_size=16, gpus=0)
    return result.system_score

def compute_token_metrics(references, predictions):
    """
    Token-level Precision, Recall, F1, F2, and Exact Match Accuracy.
    Operates on word-level sets per sentence.
    F2 weighs recall twice as much as precision (useful when missing words is 
    worse than generating extra words).
    """
    total_p, total_r, exact = 0.0, 0.0, 0
    n = len(references)

    for ref, pred in zip(references, predictions):
        ref_toks = set(ref.lower().split())
        pred_toks = set(pred.lower().split())

        if ref.strip().lower() == pred.strip().lower():
            exact += 1

        if not ref_toks or not pred_toks:
            continue

        inter = ref_toks & pred_toks
        total_p += len(inter) / len(pred_toks)
        total_r += len(inter) / len(ref_toks)

    avg_p = total_p / n
    avg_r = total_r / n
    f1 = (2 * avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
    # F2: Beta=2, weighs recall more
    f2 = (5 * avg_p * avg_r) / (4 * avg_p + avg_r) if (4 * avg_p + avg_r) > 0 else 0.0
    exact_acc = exact / n

    return avg_p, avg_r, f1, f2, exact_acc

# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_plot_confusion_matrix(references, predictions, top_n=20, save_path=None):
    """
    Adapts confusion matrix for translation via token-level alignment.
    - Selects the top N most frequent reference tokens as "classes".
    - All other tokens are grouped into "OTHER".
    - Compares reference tokens vs predicted tokens at each aligned position.
    
    TP = Predicted the correct token for that position.
    FP = Predicted a token, but the reference was something else.
    FN = Reference had a token, but the model predicted something else.
    TN = Both reference and prediction did NOT have a specific token (not meaningful in NLP).
    """
    all_ref_toks, all_pred_toks = [], []

    for ref, pred in zip(references, predictions):
        r_toks = ref.split()
        p_toks = pred.split()
        min_len = min(len(r_toks), len(p_toks))
        all_ref_toks.extend(r_toks[:min_len])
        all_pred_toks.extend(p_toks[:min_len])

    top_tokens = [t for t, _ in Counter(all_ref_toks).most_common(top_n)]
    labels = top_tokens + ["OTHER"]

    def map_tok(t):
        return t if t in top_tokens else "OTHER"

    mapped_refs  = [map_tok(t) for t in all_ref_toks]
    mapped_preds = [map_tok(t) for t in all_pred_toks]

    cm = confusion_matrix(mapped_refs, mapped_preds, labels=labels)

    # Calculate TP, FP, FN, TN
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)

    # Plot
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Token', fontsize=12)
    plt.ylabel('Reference Token', fontsize=12)
    plt.title(f'Token-Level Confusion Matrix (Top {top_n} Tokens) — 500 Samples', fontsize=14)
    plt.tight_layout()

    save_file = save_path or os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix saved → {save_file}")

    return {"TP": int(TP.sum()), "FP": int(FP.sum()), "FN": int(FN.sum()), "TN": int(TN.sum())}

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_bar_chart(metrics, save_path=None):
    """Bar chart comparing lexical vs semantic metrics."""
    names  = ['BLEU', 'chrF', 'METEOR', 'BERTScore (×100)', 'COMET (×100)']
    values = [
        metrics['BLEU'],
        metrics['chrF'],
        metrics['METEOR'],
        metrics['BERTScore_F1'] * 100,
        metrics['COMET'] * 100,
    ]

    plt.figure(figsize=(10, 6))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    bars = plt.bar(names, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=11)
    plt.ylim(0, 110)
    plt.ylabel('Score', fontsize=12)
    plt.title('ByT5 Model — Lexical vs Semantic Metric Scores', fontsize=14)
    plt.tight_layout()
    save_file = save_path or os.path.join(OUTPUT_DIR, "metrics_bar_chart.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"[*] Metrics bar chart saved → {save_file}")

def plot_precision_recall(metrics, save_path=None):
    """Scatter plot of Precision vs Recall."""
    plt.figure(figsize=(6, 6))
    plt.scatter([metrics['Recall']], [metrics['Precision']], color='crimson', s=150, zorder=5)
    plt.annotate(f"  P={metrics['Precision']:.3f}, R={metrics['Recall']:.3f}",
                 (metrics['Recall'], metrics['Precision']), fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Token-Level Precision vs Recall', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_file = save_path or os.path.join(OUTPUT_DIR, "precision_recall_chart.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"[*] Precision-Recall chart saved → {save_file}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Force CPU for now because RTX 50-series (sm_120) has compatibility issues with current PyTorch install
    device = torch.device("cpu")
    print(f"[*] FORCING DEVICE: {device} (Bypassing GPU to avoid RTX 5050 compatibility hang)")
    
    # Check model exists
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] Model not found at {MODEL_DIR}. Run byt5_train.py first.")
        return

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, TOKENIZER_DIR, device)

    # Load and sample dataset
    print(f"[*] Loading dataset from {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig").dropna(subset=[SRC_COL, TGT_COL])
    df = df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"[*] Sampled exactly {N_SAMPLES} rows for evaluation.")

    # Inference
    sources, references, predictions = generate_predictions(model, tokenizer, df, device)

    # Save predictions CSV
    pred_df = pd.DataFrame({"source_text": sources, "reference": references, "prediction": predictions})
    pred_csv_path = os.path.join(OUTPUT_DIR, "predictions_500.csv")
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"[*] Predictions saved → {pred_csv_path}")

    # ── Compute All Metrics ──────────────────────────────────────────────────
    print("[*] Computing BLEU...")
    bleu = compute_bleu(references, predictions)

    print("[*] Computing chrF...")
    chrf = compute_chrf(references, predictions)

    print("[*] Computing METEOR...")
    meteor = compute_meteor(references, predictions)

    print("[*] Computing BERTScore...")
    bs_p, bs_r, bs_f1 = compute_bert_score(references, predictions)

    print("[*] Downloading and computing COMET (this may take a minute)...")
    try:
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        comet = compute_comet(sources, references, predictions, comet_model, device)
    except Exception as e:
        print(f"[WARNING] COMET failed: {e}. Setting COMET=0.")
        comet = 0.0

    print("[*] Computing token-level metrics...")
    precision, recall, f1, f2, exact_match = compute_token_metrics(references, predictions)

    print("[*] Generating Confusion Matrix...")
    cm_stats = compute_and_plot_confusion_matrix(references, predictions)

    # ── Aggregate Results ────────────────────────────────────────────────────
    metrics = {
        "BLEU":              round(bleu, 4),
        "chrF":              round(chrf, 4),
        "METEOR":            round(meteor, 4),
        "COMET":             round(comet, 4),
        "BERTScore_P":       round(bs_p, 4),
        "BERTScore_R":       round(bs_r, 4),
        "BERTScore_F1":      round(bs_f1, 4),
        "Precision":         round(precision, 4),
        "Recall":            round(recall, 4),
        "F1":                round(f1, 4),
        "F2":                round(f2, 4),
        "Exact_Match_Accuracy": round(exact_match, 4),
        "Confusion_Matrix":  cm_stats
    }

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[*] Evaluation results saved → {json_path}")

    # Plots
    plot_metrics_bar_chart(metrics)
    plot_precision_recall(metrics)

    # ── Final Report ─────────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("       FINAL MODEL EVALUATION REPORT")
    print("-" * 50)
    print(f"BLEU:                  {metrics['BLEU']:.2f}")
    print(f"chrF:                  {metrics['chrF']:.2f}")
    print(f"METEOR:                {metrics['METEOR']:.2f}")
    print(f"COMET:                 {metrics['COMET']:.4f}")
    print(f"BERTScore (F1):        {metrics['BERTScore_F1']:.4f}")
    print(f"Precision:             {metrics['Precision']:.4f}")
    print(f"Recall:                {metrics['Recall']:.4f}")
    print(f"F1:                    {metrics['F1']:.4f}")
    print(f"F2:                    {metrics['F2']:.4f}")
    print(f"Exact Match Accuracy:  {metrics['Exact_Match_Accuracy']:.4f}")
    print(f"TP: {cm_stats['TP']} | FP: {cm_stats['FP']} | FN: {cm_stats['FN']} | TN: {cm_stats['TN']}")
    print("-" * 50)

    # ── Recommendations ──────────────────────────────────────────────────────
    print("\n[RECOMMENDATIONS]")
    if metrics['BLEU'] < 10:
        print("- BLEU is very low. But check BERTScore and COMET — if they are high, the")
        print("  model is generating semantically correct paraphrases that BLEU can't credit.")
    elif 10 <= metrics['BLEU'] < 25:
        print("- BLEU is in a reasonable range for a low-resource language. Train for more")
        print("  epochs or increase dataset size to improve further.")
    else:
        print("- BLEU is strong! This model shows excellent lexical alignment.")

    if metrics['BERTScore_F1'] > 0.85:
        print("- BERTScore is high — the model generates semantically accurate translations.")
    else:
        print("- Consider fine-tuning with a larger multilingual model like mBART or mT5.")

    if metrics['COMET'] > 0.75:
        print("- COMET score is excellent — translations are close to human quality.")
    else:
        print("- COMET is low. This is the most reliable signal — prioritize improving this.")

    print("\n[*] Evaluation complete!")

if __name__ == "__main__":
    main()
