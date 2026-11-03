# Conventional Seq2Seq Baselines for Gondi Translation

This directory provides a complete, standardized suite of **smaller conventional sequence-to-sequence (Seq2Seq) baselines trained completely from scratch** on the same parallel Gondi corpus.

These baselines satisfy academic reviewer and benchmark requirements by providing direct comparisons against larger multilingual and pretrained architectures (such as ByT5, mT5, and NLLB):

1. **BiLSTM + Attention**: 2-layer Bidirectional LSTM encoder + Bahdanau additive attention + 2-layer LSTM decoder (~24.0M parameters).
2. **GRU + Attention**: 2-layer Bidirectional GRU encoder + Bahdanau additive attention + 2-layer GRU decoder (~19.9M parameters).
3. **Standard Transformer-Small**: 4-layer Encoder / 4-layer Decoder with $d_{\text{model}}=256$, 4 heads, $d_{\text{ff}}=1024$ (~11.2M parameters).
4. **Standard Transformer-Base**: Canonical Vaswani et al. architecture: 6-layer Encoder / 6-layer Decoder with $d_{\text{model}}=512$, 8 heads, $d_{\text{ff}}=2048$ (~51.8M parameters).

All models share the identical BPE subword vocabulary (5,000 tokens) and are trained on the exact same dataset (`Datasets/gondi_training_60k.csv`) and evaluated on (`Datasets/gondi_test_40k_fixed.csv`), ensuring a strict, fair apples-to-apples comparison.

---

## Directory Structure

```text
Conventional_Baselines/
├── models/
│   ├── __init__.py
│   ├── bilstm_attention.py       # BiLSTM + Bahdanau Attention Seq2Seq
│   ├── gru_attention.py          # BiGRU + Bahdanau Attention Seq2Seq
│   └── transformer_baseline.py   # Standard Transformer (Small & Base)
├── dataset.py                    # Shared parallel dataset loader & collator
├── train.py                      # Unified training pipeline with loss & BLEU tracking
├── evaluate.py                   # Comprehensive evaluation (BLEU-1..4, chrF++, latency)
├── inference.py                  # Interactive CLI for translation
├── compare_baselines.py          # Generates comparison tables & markdown reports
├── gondi_bpe_tokenizer.py        # Shared BPE tokenizer implementation
├── gondi_bpe_model.json          # Shared 5k subword BPE model
└── README.md                     # Documentation
```

---

## Architectural Details

| Baseline Model | Paradigm | Encoder | Decoder | Attention Mechanism | Parameters |
|---|---|---|---|---|---|
| **BiLSTM + Attention** | Recurrent (LSTM) | 2-layer BiLSTM ($H=512$, $E=256$) | 2-layer LSTM ($H=512$) | Bahdanau Additive Attention | ~24.0M |
| **GRU + Attention** | Recurrent (GRU) | 2-layer BiGRU ($H=512$, $E=256$) | 2-layer GRU ($H=512$) | Bahdanau Additive Attention | ~19.9M |
| **Transformer-Small** | Attention-only | 4 layers ($d=256, h=4, d_{\text{ff}}=1024$) | 4 layers ($d=256, h=4, d_{\text{ff}}=1024$) | Scaled Dot-Product (Self & Cross) | ~11.2M |
| **Transformer-Base** | Attention-only | 6 layers ($d=512, h=8, d_{\text{ff}}=2048$) | 6 layers ($d=512, h=8, d_{\text{ff}}=2048$) | Scaled Dot-Product (Self & Cross) | ~51.8M |

---

## Quickstart & Training

### 1. Training from Scratch

To train any baseline on the Gondi parallel corpus (`Gondi -> Hindi` or `Gondi -> English`):

```bash
# 1. Train BiLSTM + Attention
python train.py --model bilstm --tgt_lang Hindi --epochs 30 --batch_size 32

# 2. Train GRU + Attention
python train.py --model gru --tgt_lang Hindi --epochs 30 --batch_size 32

# 3. Train Standard Transformer-Small
python train.py --model transformer_small --tgt_lang Hindi --epochs 30 --batch_size 32

# 4. Train Standard Transformer-Base
python train.py --model transformer_base --tgt_lang Hindi --epochs 30 --batch_size 32
```

Checkpoints and training histories are automatically saved to `checkpoints/`.

---

## Evaluation Benchmark

Evaluate a saved checkpoint on the held-out test set (`Datasets/gondi_test_40k_fixed.csv`):

```bash
python evaluate.py --model bilstm --checkpoint checkpoints/best_bilstm_hindi.pt --num_samples 1000
python evaluate.py --model gru --checkpoint checkpoints/best_gru_hindi.pt --num_samples 1000
python evaluate.py --model transformer_small --checkpoint checkpoints/best_transformer_small_hindi.pt --num_samples 1000
python evaluate.py --model transformer_base --checkpoint checkpoints/best_transformer_base_hindi.pt --num_samples 1000
```

Outputs include:
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4** (Cumulative)
- **chrF++**
- **Inference latency** (ms/sentence) and **throughput** (tokens/sec)
- Qualitative translation examples

---

## Interactive Translation

Translate custom Gondi sentences directly:

```bash
# Interactive prompt:
python inference.py --model gru --checkpoint checkpoints/best_gru_hindi.pt

# Direct CLI translation:
python inference.py --model transformer_small --checkpoint checkpoints/best_transformer_small_hindi.pt --text "मना नाव राहुल"
```

---

## Comparative Benchmark Suite

To compile results across all 4 baselines into a consolidated paper-ready table:

```bash
python compare_baselines.py --tgt_lang hindi
```

This outputs both `baseline_comparison.md` and `baseline_comparison.csv`.

