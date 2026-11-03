# BiLSTM Seq2Seq with Bahdanau Attention (Gondi Translation)

A sequence-to-sequence baseline using a **Bidirectional LSTM Encoder** and an **LSTM Decoder** equipped with **Bahdanau Additive Attention**.

---

## Architecture Specifications

- **Encoder**: 2-layer Bidirectional LSTM ($H=512$, $E=256$, Dropout=$0.5$)
  - Forward and backward hidden and cell states are projected to $H=512$.
- **Attention**: Bahdanau additive alignment model:
  $$e_{t,i} = v^T \tanh(W [s_{t-1}; h_i])$$
- **Decoder**: 2-layer LSTM ($H=512$). Input is concatenated embedding and context vector.
- **Vocabulary**: 5,000 BPE subword units shared across source and target languages.
- **Parameters**: ~24.0M trainable parameters.

---

## Scripts & Usage

- **Training**:
  ```bash
  python train.py
  ```

- **Interactive Inference**:
  ```bash
  python inference.py
  ```

- **Evaluation Benchmark**:
  ```bash
  python evaluate.py --checkpoint best_gondi_model.pt --num_samples 1000
  ```

- **Confusion Matrix**:
  ```bash
  python generate_cm.py
  ```

