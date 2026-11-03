# Standard Transformer Baselines (Small and Base) — Gondi Translation

Implements standard sequence-to-sequence Transformers (Vaswani et al., 2017) trained completely from scratch on the Gondi parallel corpus.

---

## Architectural Configurations

| Configuration | Encoder Layers | Decoder Layers | $d_{\text{model}}$ | Heads | $d_{\text{ff}}$ | Dropout | Trainable Params |
|---|---|---|---|---|---|---|---|
| **Transformer-Small** | 4 | 4 | 256 | 4 | 1024 | 0.1 | ~11.2M |
| **Transformer-Base**  | 6 | 6 | 512 | 8 | 2048 | 0.1 | ~51.8M |

- **Positional Encoding**: Sinusoidal positional encodings.
- **Attention**: Scaled dot-product multi-head attention with causal triangular masks in decoder and padding masks.
- **Vocabulary**: 5,000 subword units (BPE).

---

## Scripts & Usage

- **Training**:
  ```bash
  # Train Transformer-Small:
  python train.py --model_size small --tgt_lang Hindi --epochs 30

  # Train Transformer-Base:
  python train.py --model_size base --tgt_lang Hindi --epochs 30
  ```

- **Interactive Inference**:
  ```bash
  python inference.py
  ```

- **Evaluation**:
  ```bash
  python evaluate.py --model_size small --num_samples 1000
  ```

- **Confusion Matrix**:
  ```bash
  python generate_cm.py --model_size small
  ```

