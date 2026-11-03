# Technical Report: Architectural Specification and Parameter Analysis of the Custom T5 Gondi Model

---

## Executive Summary
This report presents the architectural design, layer-by-layer mathematical parameter calculations, and a comparative analysis of the custom T5 model (`t5_gondi_model`) developed for Gondi-English/Hindi translation. By optimizing vocabulary bounds and scaling down layer dimensions, we achieved a highly efficient **10.79 Million parameter** model designed to train rapidly on low-resource parallel corpora while remaining highly resilient to overfitting.

---

## 1. Architectural Overview & Configuration
The model is built using the standard T5 (Text-to-Text Transfer Transformer) encoder-decoder architecture, initialized from scratch (randomly initialized weights) rather than loaded from pre-trained checkpoints. 

The structural hyperparameters of the network are defined as follows:

| Hyperparameter | Description | Value |
| :--- | :--- | :--- |
| $V$ | Vocabulary Size | $1,157$ |
| $d_{model}$ | Model / Hidden Representation Dimension | $256$ |
| $d_{ff}$ | Feed-Forward Layer Dimension | $1,024$ |
| $h$ | Attention Heads | $8$ |
| $d_{kv}$ | Key-Value Projection Dimension | $64$ |
| $N_{enc}$ | Encoder Blocks | $4$ |
| $N_{dec}$ | Decoder Blocks | $4$ |

---

## 2. Layer-by-Layer Mathematical Breakdown

The total parameter count ($P_{total}$) of the network is computed as the sum of the Shared Word Embeddings ($P_{embed}$), Encoder Parameters ($P_{enc}$), and Decoder Parameters ($P_{dec}$):
$$P_{total} = P_{embed} + P_{enc} + P_{dec}$$

### A. Shared Word Embeddings ($P_{embed}$)
The embedding layer maps discrete token IDs into dense vectors of dimension $d_{model}$. In the T5 architecture, the input embeddings and output projection layer (Language Modeling Head) share/tie weights to save memory.
$$P_{embed} = V \times d_{model}$$
$$P_{embed} = 1,157 \times 256 = \mathbf{296,192\text{ parameters}}$$

---

### B. Encoder Architecture ($P_{enc}$)
The encoder is composed of $N_{enc} = 4$ identical blocks. Each block is partitioned into two main sub-layers: **Self-Attention** and a **Feed-Forward Network (FFN)**. 

#### I. Sub-Layer 0: Multi-Head Self-Attention
* **Projection Matrices ($W_Q, W_K, W_V, W_O$)**:
  Queries, Keys, and Values project from $d_{model}$ (256) to the attention workspace ($h \times d_{kv} = 512$):
  * $W_Q \text{ weight shape } [512, 256] = 131,072$
  * $W_K \text{ weight shape } [512, 256] = 131,072$
  * $W_V \text{ weight shape } [512, 256] = 131,072$
  * $W_O \text{ weight shape } [256, 512] = 131,072$
  $$\text{Projection subtotal} = 131,072 \times 4 = 524,288\text{ parameters}$$
* **RMS Normalization (Scale weight)**:
  * $\text{RMSNorm shape } [256] = 256\text{ parameters}$
* **Relative Position Bias Table** *(Instantiated in Encoder Block 0 only)*:
  * $\text{Bias matrix shape } [32\text{ buckets}, 8\text{ heads}] = 256\text{ parameters}$

#### II. Sub-Layer 1: Dense-Relu-Dense FFN
* **Feed-Forward Weights ($W_i, W_o$)**:
  Projects representations to a higher dimension and back ($d_{model} \to d_{ff} \to d_{model}$):
  * $W_i \text{ weight shape } [1024, 256] = 262,144$
  * $W_o \text{ weight shape } [256, 1024] = 262,144$
  $$\text{FFN subtotal} = 262,144 \times 2 = 524,288\text{ parameters}$$
* **RMS Normalization (Scale weight)**:
  * $\text{RMSNorm shape } [256] = 256\text{ parameters}$

#### III. Encoder Global Layer Norm
* $\text{Encoder final RMSNorm shape } [256] = 256\text{ parameters}$

#### IV. Encoder Summation
$$P_{enc} = N_{enc} \times \left( \text{Self-Attn Projections} + \text{RMSNorm}_1 + \text{FFN Projections} + \text{RMSNorm}_2 \right) + \text{RelBias} + \text{FinalNorm}$$
$$P_{enc} = 4 \times \left( 524,288 + 256 + 524,288 + 256 \right) + 256 + 256$$
$$P_{enc} = 4 \times 1,049,088 + 512 = \mathbf{4,196,864\text{ parameters}}$$

---

### C. Decoder Architecture ($P_{dec}$)
The decoder consists of $N_{dec} = 4$ identical blocks. Each block contains three sub-layers: **Self-Attention**, **Encoder-Decoder Cross-Attention**, and a **Feed-Forward Network (FFN)**.

#### I. Sub-Layer 0: Multi-Head Self-Attention
* Same as Encoder (Self-Attention Projections + RMSNorm) = $524,288 + 256 = 524,544\text{ parameters}$
* **Relative Position Bias Table** *(Instantiated in Decoder Block 0 only)*:
  * $\text{Bias matrix shape } [32\text{ buckets}, 8\text{ heads}] = 256\text{ parameters}$

#### II. Sub-Layer 1: Encoder-Decoder Cross-Attention (EncDecAttention)
* **Projection Matrices**: Queries project from decoder states; Keys/Values project from encoder outputs.
  * $W_Q, W_K, W_V, W_O \text{ shape equivalent to Self-Attention} = 524,288\text{ parameters}$
* **RMS Normalization (Scale weight)**:
  * $\text{RMSNorm shape } [256] = 256\text{ parameters}$

#### III. Sub-Layer 2: Dense-Relu-Dense FFN
* Same as Encoder FFN (FFN Projections + RMSNorm) = $524,288 + 256 = 524,544\text{ parameters}$

#### IV. Decoder Global Layer Norm
* $\text{Decoder final RMSNorm shape } [256] = 256\text{ parameters}$

#### V. Decoder Summation
$$P_{dec} = N_{dec} \times \left( \text{Self-Attn} + \text{Cross-Attn} + \text{FFN} + 3 \times \text{RMSNorm} \right) + \text{RelBias} + \text{FinalNorm}$$
$$P_{dec} = 4 \times \left( 524,288 + 524,288 + 524,288 + 768 \right) + 256 + 256$$
$$P_{dec} = 4 \times 1,573,632 + 512 = \mathbf{6,295,040\text{ parameters}}$$

---

### D. Grand Total Parameter Synthesis
Summing the individual subsystems yields the exact total parameters:
$$P_{total} = P_{embed} + P_{enc} + P_{dec}$$
$$P_{total} = 296,192 + 4,196,864 + 6,295,040 = \mathbf{10,788,096\text{ parameters}}$$

---

## 3. Comparative Architectural Analysis

The following table contrasts our scaled down custom model configuration with standard Google pre-trained T5 configurations:

| Metric | Google T5-Small | Google T5-Base | Custom T5 Gondi Model |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | **60.5 Million** | **220 Million** | **10.79 Million** |
| **Vocabulary Size ($V$)** | $32,128$ | $32,128$ | $1,157$ |
| **Model Dimension ($d_{model}$)** | $512$ | $768$ | $256$ |
| **Feed-Forward Dimension ($d_{ff}$)** | $2,048$ | $3,072$ | $1,024$ |
| **Attention Heads ($h$)** | $8$ | $12$ | $8$ |
| **Encoder / Decoder Layers** | $6\text{ / } 6$ | $12\text{ / } 12$ | $4\text{ / } 4$ |
| **Embedding Size** | $16.45\text{M}$ | $24.67\text{M}$ | $0.29\text{M}$ |
| **Weights Initialization** | Pre-trained | Pre-trained | Scratch (Random) |

---

## 4. Optimization Rationale

The parameters were optimized using three core structural adjustments:

1. **Vocabulary Shrinkage**:
   Standard multilingual T5 models require a vocab of 32,128 to cover dozens of languages. Because our tokenizer is domain-trained on a low-resource corpus of Gondi-English/Hindi texts, BPE converges on **1,157** tokens. This saves over **$16.1$ Million parameters** in the embedding layer alone.
2. **Dimension Narrowing**:
   Reducing the representation vector ($d_{model}$) from 512 to 256 and FFN hidden dimension ($d_{ff}$) from 2048 to 1024 reduces the size of all weight matrices quadratically, reducing linear layer sizes by a factor of 4.
3. **Layer Depth Reduction**:
   Moving from 6 blocks to 4 blocks in both the encoder and decoder decreases computational complexity and training time by roughly 33%, allowing for stable convergence on a 60,000-sentence translation dataset.
