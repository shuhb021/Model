# Custom Asymmetric Multilingual NMT
### Gondi ↔ Hindi ↔ English | ~137M Parameters

Research-grade Neural Machine Translation for low-resource tribal Indic languages,
as per the formal architecture specification.

---

## Architecture Summary

| Component | Specification |
|---|---|
| **Parameters** | ~137.25M (full) / ~3.1M (LoRA fine-tune) |
| **Encoder** | 8 layers — deeper for cross-lingual feature extraction |
| **Decoder** | 4 layers — shallow to reduce autoregressive error |
| **Attention Heads** | 8 (d_k = 64) |
| **Hidden Dim** | d_model = 512 |
| **FFN** | Gated-GeLU (d_ff = 2048) |
| **Positional Encoding** | T5-style Relative Bias (shared across layers, per-head learned) |
| **Vocabulary** | 32,000 (shared multilingual BPE) |
| **Dropout** | 0.15 (scaled for low-resource) |

---

## Project Structure

```
nmt_model/
├── main.py                        # Entry point
├── requirements.txt
├── configs/
│   └── config.yaml                # Full model + training config
├── model/
│   ├── architecture.py            # AsymmetricNMT, Encoder, Decoder, GatedGeLUFFN, RelativePositionBias
│   └── lora.py                    # LoRA adapter injection (rank=16, alpha=32)
├── data/
│   ├── tokenizer.py               # Multilingual BPE tokenizer wrapper + language tags
│   └── pipeline.py                # Data preprocessing pipeline + temperature sampler
├── training/
│   └── trainer.py                 # NMTTrainer with curriculum learning stages
└── inference/
    ├── decoder.py                 # MBR / Beam / Greedy decoding + chrF metric
    └── translation_memory.py      # Dual TM retrieval (Levenshtein + TF-IDF/FAISS)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify architecture

```bash
python main.py --mode verify
```

Expected output:
```
📊 Parameter Count:
   Total trainable parameters :     137,248,xxx
   Approx (millions)          :          137.xx M
✅ Architecture verification PASSED
```

### 3. Run tests

```bash
python main.py --mode test
```

### 4. Train SentencePiece tokenizer

Collect your parallel corpus into a single file (one sentence per line, all languages):

```bash
cat data/gondi_hindi.txt data/gondi_english.txt data/hindi_english.txt > data/combined_corpus.txt
```

Then train the BPE model:

```bash
python data/tokenizer.py   # prints the spm.SentencePieceTrainer.Train() command
```

### 5. Preprocess data

```python
from data.pipeline import DataPipeline

pipeline = DataPipeline(output_dir="data/processed")
pairs = list(zip(open("raw/gondi.txt").read().splitlines(),
                 open("raw/hindi.txt").read().splitlines()))
pipeline.run(pairs, src_lang="gondi", tgt_lang="hindi")
```

### 6. Train

```python
from model.architecture import AsymmetricNMT
from model.lora import apply_lora_to_model
from training.trainer import NMTTrainer

model = AsymmetricNMT()
model = apply_lora_to_model(model, rank=16, alpha=32)   # Stage 3 only

trainer = NMTTrainer(
    model=model,
    train_loaders={"hi_en": hi_en_loader,
                   "multilingual": multi_loader,
                   "gondi": gondi_loader},
    val_loader=val_loader,
)
trainer.run_curriculum()
```

### 7. Translate

```bash
# Interactive
python main.py --mode translate --src_lang gondi --tgt_lang hindi

# File batch
python main.py --mode translate \
    --src_lang gondi --tgt_lang hindi \
    --input_file sentences.txt \
    --output_file translations.txt \
    --checkpoint checkpoints/best_model.pt
```

---

## Key Design Decisions

### Why Asymmetric (8 Enc / 4 Dec)?
Gondi requires deep encoding to capture agglutinative morphology and postpositional syntax.
A shallow decoder reduces autoregressive compounding errors on small corpora.

### Why Gated-GeLU FFN?
The gating pathway dynamically filters morphological suffixes (e.g., `-tor`, `-lasi` in Gondi),
improving representation of agglutinative word structures over standard ReLU/GeLU FFNs.

### Why T5 Relative Position Bias?
Gondi has free word order. Absolute position embeddings break down when phrase positions shift.
Relative bias captures structural dependencies regardless of word order.

### Why MBR over Beam Search?
Beam search hallucinates and language-switches on low-resource pairs.
MBR with K=16 samples selects the highest-consensus candidate, suppressing statistical outliers.

### Why LoRA (not full fine-tune)?
Full fine-tuning on 600K tribal pairs causes catastrophic forgetting of multilingual representations.
LoRA adapters freeze base parameters and update only ~3.1M parameters (2.3% of total).

---

## Language Codes

| Language | Tag | Script |
|---|---|---|
| Gondi | `__gon_Latn__` | Latin (transliterated) |
| Hindi | `__hin_Deva__` | Devanagari |
| English | `__eng_Latn__` | Latin |

---

## References

- Vaswani et al. (2017). *Attention Is All You Need*
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with T5*
- Noam et al. (2019). *LoRA: Low-Rank Adaptation of Large Language Models*
- Costa-jussà et al. (2022). *No Language Left Behind (NLLB-200)*
- Müller & Sennrich (2021). *Understanding the Properties of Minimum Bayes Risk Decoding*
