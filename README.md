# Gondi Neural Machine Translation & Conventional Baselines

A comprehensive codebase for low-resource tribal language translation (**Gondi ↔ Hindi ↔ English**), containing state-of-the-art multilingual pretrained architectures, custom asymmetric NMT models, and smaller conventional sequence-to-sequence baselines trained completely from scratch.

---

## 📁 Repository Structure

```text
├── Conventional_Baselines/     # Unified baseline suite (BiLSTM, GRU, Transformer Small/Base)
│   ├── models/                 # PyTorch model definitions
│   ├── dataset.py              # Unified BPE dataset loader & collator
│   ├── train.py                # Unified baseline trainer with BLEU & loss tracking
│   ├── evaluate.py             # Benchmark evaluation (BLEU-1..4, chrF++, latency)
│   ├── inference.py            # Interactive translation CLI
│   └── compare_baselines.py    # Benchmark table generator
│
├── Byt5 Model/                 # ByT5 multilingual translation models
│   ├── Scripts/                # Training, evaluation, inference, and vocab ablation
│   └── Tokenizer_Files/        # BPE tokenizers and dynamic trainer (train_bpe_tokenizer.py)
│
├── GRU_Model/                  # Standalone GRU + Bahdanau Attention baseline
├── LSTM_Model/                 # Standalone BiLSTM + Bahdanau Attention baseline
├── Transformer_Model/          # Standalone Standard / Vanilla Transformer (Small & Base)
├── T5 MODEL/                   # T5 conditional generation model & scripts
├── NLLB/                       # NLLB multilingual model implementation & retrieval
├── nmt_model/                  # Custom Asymmetric Multilingual NMT Architecture (~137M)
├── ML_Model_mT5_base/          # mT5/ByT5 training and inference scripts
├── Datasets/                   # Test datasets, dictionaries, and processing scripts
│   ├── gondi_test_40k_fixed.csv
│   ├── extract_60k.py
│   ├── process_dataset.py
│   └── *.pdf (Dictionaries & Grammar references)
│
├── others/                     # Parameter breakdowns and analysis documentation
├── run_all_experiments.py      # Master experiment runner (P1, P2, P3 reviewer tasks)
└── .gitignore                  # Strict exclusion of training datasets and checkpoints
```

---

## ⚡ Models & Architectures

| Model | Category | Architecture / Parameters | Script / Entrypoint |
|---|---|---|---|
| **Custom Asymmetric NMT** | Tribal NMT | 8-Enc / 4-Dec, T5 Relative Bias, Gated-GeLU (~137M) | `nmt_model/main.py` |
| **ByT5 Gondi** | Pretrained / Subword | 8-Enc / 4-Dec, Gated-GeLU, configurable BPE | `Byt5 Model/Scripts/byt5_train.py` |
| **T5 Model** | Pretrained | T5 conditional generation pipeline | `T5 MODEL/T5_Model/t5_train.py` |
| **NLLB** | Multilingual SOTA | Multilingual encoder-decoder with retrieval | `NLLB/train.py` |
| **Vanilla Transformer** | Conventional Baseline | 6-Enc / 6-Dec, 8 heads, $d_{\text{model}}=512$ (~51.8M) | `Transformer_Model/vanilla_transformer.py` |
| **Transformer-Small** | Conventional Baseline | 4-Enc / 4-Dec, 4 heads, $d_{\text{model}}=256$ (~11.2M) | `Conventional_Baselines/train.py --model transformer_small` |
| **BiLSTM + Attention** | Conventional Baseline | 2-layer BiLSTM ($H=512$) + Bahdanau Attention (~24.0M) | `Conventional_Baselines/train.py --model bilstm` |
| **GRU + Attention** | Conventional Baseline | 2-layer BiGRU ($H=512$) + Bahdanau Attention (~19.9M) | `Conventional_Baselines/train.py --model gru` |

---

## 🚀 Running Experiments

All experiment tasks can be inspected or executed via `run_all_experiments.py`:

```bash
# Check planned commands and matrix without running (dry run):
python run_all_experiments.py --dry_run

# P1: ByT5 with different BPE vocabulary sizes (e.g. 1157, 2000, 4000, 8000):
python run_all_experiments.py --run_p1 --vocabs 1157 2000 4000 --epochs 5

# P2: Vanilla Transformer on same data & BPE tokenizer:
python run_all_experiments.py --run_p2 --epochs 30

# P3: BiLSTM + Attention and GRU + Attention baselines:
python run_all_experiments.py --run_p3 --epochs 30

# Run all baseline comparisons:
python run_all_experiments.py --all --epochs 30
```

---

## 📊 Evaluation & Benchmarking

Evaluate any trained baseline on the held-out test set (`Datasets/gondi_test_40k_fixed.csv`):

```bash
cd Conventional_Baselines

# Evaluate BLEU-1..4, chrF++, and latency:
python evaluate.py --model gru --checkpoint checkpoints/best_gru_hindi.pt --num_samples 1000

# Interactive real-time translation:
python inference.py --model transformer_small --checkpoint checkpoints/best_transformer_small_hindi.pt

# Generate comparison markdown table:
python compare_baselines.py --tgt_lang hindi
```

---

## 🔒 Data Policy

This repository includes code, scripts, tokenizers, and benchmark test sets (`Datasets/gondi_test_40k_fixed.csv`, dictionaries). Raw large training corpora (`gondi_training_60k.csv`, `gondi_training_dataset.csv`, `gondi_dataset_full.csv`) and model checkpoint files (`.pt`, `.bin`) are explicitly excluded via `.gitignore`.
