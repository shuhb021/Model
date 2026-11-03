# Custom Asymmetric Multilingual Neural Machine Translation (NMT) Architecture for Low-Resource Tribal Indic Languages

## 1. Complete Architecture Overview

This document presents the formal architectural specification for a custom, research-grade asymmetric multilingual Neural Machine Translation (NMT) model. The model is specifically optimized for low-resource tribal Indic languages (Gondi ↔ Hindi, Gondi ↔ English) with target scaling calibrated for approximately $600,000+$ sentence pairs. 

### 1.1 Core Design Philosophy
Low-resource machine translation for highly agglutinative tribal languages suffers from severe overfitting and vocabulary sparsity when using symmetric high-capacity architectures (like NLLB-200 or mBART). To mitigate these challenges, our architecture implements three key design shifts:
1. **Asymmetric Layout**: Placing higher representation-learning capacity in the Encoder (8 layers) and lower capacity in the Decoder (4 layers). Deeper encoders excel at structural extraction and cross-lingual alignment, whereas shallow decoders reduce autoregressive compounding error and decrease training and inference latencies.
2. **Relative Positional Encoding**: Replacing absolute position embeddings with T5-style relative position bias, enabling sequence length generalization and robust handling of free-word-order tribal syntax.
3. **Morphology-Aware Gated FFNs**: Utilizing Gated-GeLU feed-forward networks (FFNs) to model morphology-rich agglutinative suffixes.

### 1.2 Mathematical Formulations

Given a source sequence $X = (x_1, \dots, x_M)$ and a target sequence $Y = (y_1, \dots, y_N)$, the translation task models the conditional probability:
$$P(Y|X; \theta) = \prod_{i=1}^N P(y_i | y_{<i}, X; \theta)$$

#### Transformer Attention with Relative Position Bias
In standard Transformers, absolute position embeddings are added to input token embeddings. In our custom architecture, relative position biases are injected directly into the self-attention matrix. For query $Q$, key $K$, and value $V$ of dimension $d_k$, the attention matrix $A$ for head $h$ is formulated as:

$$A_{ij} = \frac{(Q_i) (K_j)^T}{\sqrt{d_k}} + \beta_{ij}$$

$$\text{Attention}(Q, K, V) = \text{Softmax}(A) V$$

where $\beta_{ij}$ is a scalar representing the relative position bias between positions $i$ and $j$. The bias values $\beta_{ij}$ are shared across all layers but learned per attention head, mapped to buckets using a logarithmic scale to handle long sequences efficiently.

#### Gated-GeLU Feed-Forward Network
Standard Transformers use a standard ReLU or GeLU activation in their FFN layers:
$$\text{FFN}_{\text{standard}}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2$$

Our model employs a Gated Gated-Linear Unit (GLU) variant with GeLU activation, which acts as a dynamic router for morphology features:
$$\text{FFN}_{\text{Gated-GeLU}}(x) = \left( \text{GeLU}(x W_g) \otimes (x W_1) \right) W_2$$

where $\otimes$ denotes the element-wise (Hadamard) product. The gate pathway $\text{GeLU}(x W_g)$ dynamically filters input morphemes, enhancing representation learning for complex word structures.

### 1.3 System Architecture Diagram

```
                [Source Tokens: Gondi/Hindi/English]
                                |
                   [Shared Multilingual Vocab]
                                |
                   [Word Embeddings (d_model=512)]
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   [Encoder Self-Attention]              [Relative Pos Bias (T5)]
  (Multi-head, H=8, d_k=64)                        |
             |                                     |
             +------------------+------------------+
                                |
                                v
                      [LayerNorm & Residual]
                                |
                   [Gated-GeLU FFN (d_ff=2048)]
                                |
                      [LayerNorm & Residual]
                                |
                    [Repeat for L_enc = 8 Layers]
                                |
                                v  (Encoder Key-Value States)
             ===================*===================
                                |
                                v
                [Target Tokens: [LANG_TAG] y_{<i}]
                                |
                   [Word Embeddings (d_model=512)]
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   [Decoder Masked Self-Attn]            [Relative Pos Bias (T5)]
  (Multi-head, H=8, d_k=64)                        |
             |                                     |
             +------------------+------------------+
                                |
                                v
                      [LayerNorm & Residual]
                                |
                   [Encoder-Decoder Cross-Attn] <--- (Encoder States)
                                |
                      [LayerNorm & Residual]
                                |
                   [Gated-GeLU FFN (d_ff=2048)]
                                |
                      [LayerNorm & Residual]
                                |
                    [Repeat for L_dec = 4 Layers]
                                |
                                v
                        [Linear Head]
                                |
                                v
                    [Softmax over Vocab Size]
                                |
                                v
                    [Output Translation Tokens]
```

---

## 2. Parameter Breakdown and Model Configuration

### 2.1 Scale Argumentation
For a parallel corpus of ~600,000 lines, standard NLLB-200 checkpoints (ranging from 600M to 3.3B parameters) are highly prone to catastrophic forgetting or severe overfitting when fine-tuned directly on low-resource pairs. Conversely, tiny models fail to capture cross-lingual transfer semantics. 

Our custom configuration targets **~137M parameters**, placing it in the optimal low-resource scaling regime: large enough to capture multilingual syntactic transfer, yet small enough to avoid overfitting and be trainable on consumer/workstation RTX GPUs.

### 2.2 Model Configuration Table

| Configuration Parameter | Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 32,000 | Shared multilingual SentencePiece/BPE vocabulary |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | Word embedding and hidden representation size |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 2,048 | Inner dimension of the Gated-GeLU layers |
| **Encoder Layers ($L_{\text{enc}}$)** | 8 | Deeper encoder for robust feature representation |
| **Decoder Layers ($L_{\text{dec}}$)** | 4 | Shallow decoder to accelerate autoregressive steps |
| **Attention Heads ($H$)** | 8 | Number of heads in multi-head attention |
| **Head Dimension ($d_k$)** | 64 | Dimension per attention head ($d_{\text{model}} / H$) |
| **Activation Function** | Gated-GeLU | Nonlinearity in Feed-Forward Networks |
| **Positional Encoding** | T5 Relative Bias | Non-parametric positional representation |
| **Dropout** | 0.15 | Regularization rate (scaled up for low-resource training) |
| **Max Sequence Length** | 256 | Maximum tokens per source/target sentence |
| **Optimizer** | AdamW | $\beta_1=0.9, \beta_2=0.98, \epsilon=1\text{e-}8$ |
| **Scheduler** | Cosine with Warmup | Linear warmup of 4,000 steps, decaying to 1e-6 |
| **Training Precision** | Mixed Precision (FP16/BF16) | Native PyTorch AMP (Automatic Mixed Precision) |
| **Batch Size / Acc. Steps** | 32 / 4 | Effective batch size of 128 sentences |
| **Estimated Parameters** | ~137,248,000 | Calculated parameter size (including embeddings) |

---

## 3. Component-Level Design Specification

### 3.1 Tokenization Design
Agglutinative tribal languages like Gondi rely heavily on word compounding and affixation (e.g., nouns inflected for cases, verbs inflected for tense, gender, and number). Standard character-level tokenizers (like ByT5) suffer from slow inference speeds and high computational overhead, while word-level tokenizers result in massive out-of-vocabulary (OOV) rates.

*   **Tokenizer Choice**: Joint Multilingual BPE (Byte-Pair Encoding) with a vocabulary of 32,000.
*   **Language-Specific Tags**: We prepend specific language identifiers to coordinate translation directions:
    - `__gon_Latn__` for Gondi (Latin script)
    - `__hin_Deva__` for Hindi (Devanagari script)
    - `__eng_Latn__` for English (Latin script)
*   **Agglutinative Suffix Processing**: By training BPE on a joint corpus, suffixes like `-tor</w>` (denoting agentive/masculine nouns in Gondi) and `-lasi</w>` (inflections) are preserved as distinct subword tokens, ensuring morpho-syntactic transfer from Hindi case markers to Gondi postpositions.
*   **Transliteration Handling**: Script-level normalization is performed during tokenization preprocessing, converting potential Devanagari-written Gondi sentences into uniform Latin representations to maximize parameter efficiency.

### 3.2 Gated-GeLU Feed-Forward Networks

The inner feed-forward network uses two projection weights for gating: $W_1$ and $W_g$, followed by a projection $W_2$.

#### Mathematical Layout:
$$\text{Intermediate}_1 = x W_1 \quad (W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}})$$
$$\text{Intermediate}_g = \text{GeLU}(x W_g) \quad (W_g \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}})$$
$$\text{Gated} = \text{Intermediate}_1 \odot \text{Intermediate}_g$$
$$\text{Output} = \text{Gated} W_2 \quad (W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}})$$

```
                   Input Tensor [Batch, Seq, d_model]
                                |
             +------------------+------------------+
             |                                     |
             v (Linear W_1)                        v (Linear W_g)
    [Batch, Seq, d_ff]                    [Batch, Seq, d_ff]
             |                                     |
             |                                     v (GeLU)
             |                            [Batch, Seq, d_ff]
             |                                     |
             +------------------+------------------+
                                |
                                v (Hadamard / Element-wise Product)
                       [Batch, Seq, d_ff]
                                |
                                v (Linear W_2)
                    Output Tensor [Batch, Seq, d_model]
```

### 3.3 Attention Flow
Multi-Head Attention processes information in parallel sub-spaces. The relative position bias adds a dynamic distance penalty:

```
 Query Vector (Q)     Key Vector (K)      Value Vector (V)
       |                   |                     |
       +------>( Q @ K.T )<--+                     |
                 |                               |
                 v / sqrt(d_k)                   |
           [Scale Matrix]                        |
                 |                               |
                 v + Relative Position Bias (T5) |
          [Biased Scores]                        |
                 |                               |
                 v Softmax                       |
         [Attention Weights]                     |
                 |                               |
                 v @ V <-------------------------+
                 |
                 v (Concat Heads & Out-Project)
          [Output Embeddings]
```

### 3.4 Relative Positional Encoding vs. Absolute Positional Embeddings
In absolute positional encodings, an embedding vector $p_i$ is mapped to index $i$ and added to token embeddings:
$$h_i = e_i + p_i$$

This scheme breaks down in free-word-order tribal languages like Gondi, where phrases can shift positions without altering the core semantic dependencies. 
In contrast, T5-style Relative Positional Encodings learn a relative position bias parameter $\beta(i - j)$ shared across layers. This bias calculates distance differences $i - j$ and assigns them to bucket ranges, allowing the model to focus on structural syntax relative to neighboring words, facilitating zero-shot sequence length generalization during decoding.

---

## 4. Retrieval-Augmented Translation Memory (TM)

To reduce hallucinations in low-resource settings, we couple the neural model with a Retrieval-Augmented Translation Memory layer. 

### 4.1 Retrieval Pipeline Architecture

```
               [Query Source Sentence]
                          |
             +------------+------------+
             |                         |
             v                         v
     (Fuzzy Matching)          (Dense Retrieval)
     Levenshtein Dist.       Sentence-Transformer
             |                         |
             v                         v
       [Top-K Candidates]        [Top-K Candidates]
             |                         |
             +------------+------------+
                          |
                          v
                 [Reranking Strategy]
              (Length + Cosine Similarity)
                          |
                          v
              [Fusion Context Generation]
        Prefix format: "TM: {Src} = {Tgt} <SEP> {Query}"
                          |
                          v
             [Decoder Generation Target]
```

### 4.2 Fusion Details
1. **Fuzzy Search**: A fast character-level Levenshtein similarity scorer checks the local dictionary/training-memory for source sentence overlap.
2. **Semantic Similarity**: A sentence transformer model converts the source sentence to an embedding, executing a similarity check against a pre-built FAISS index of the 600K parallel corpus.
3. **Prompt Prefix Fusion**: The top retrieval result is formatted as a prefix:
   `[TM_START] gondi_src_match [TM_SEP] hindi_tgt_match [TM_END] [LANG_TAG] query_source`
   The model is trained to copy matching structures when similarity is high ($>0.85$) or translate natively when similarity is low.

---

## 5. Multilingual Training & Transfer Strategy

To bridge the gap for Gondi, we leverage cross-lingual transfer from Hindi (a high-resource Indic language) and English. 

### 5.1 Curriculum Learning Stages
*   **Stage 1: High-Resource Warmup**: Pretrain the architecture on Hindi ↔ English data (from our 600K dataset split) to stabilize cross-attention alignments and representation parameters.
*   **Stage 2: Multilingual Adaptation**: Train Gondi ↔ Hindi, Gondi ↔ English, and Hindi ↔ English jointly using temperature-based sampling.
*   **Stage 3: Gondi Specialization**: Freeze the shared embeddings and lower layers, using parameter-efficient fine-tuning (PEFT/LoRA) on Gondi-specific pairs.

### 5.2 Temperature-Based Batch Sampling
To prevent high-resource translation directions (Hindi-English) from dominating the gradients over low-resource directions (Gondi-Hindi, Gondi-English), we compute the sampling probability for dataset $i$ using temperature $T = 1.6$:

$$P_i = \frac{(N_i)^{1/T}}{\sum_j (N_j)^{1/T}}$$

where $N_i$ is the number of sentence pairs in dataset $i$.

---

## 6. Parameter-Efficient Fine-Tuning (PEFT) & Optimization

Rather than performing full fine-tuning, which alters the pretrained multilingual representations and leads to catastrophic forgetting, we freeze the base parameters $\theta_0$ and inject Low-Rank Adaptation (LoRA) adapters into the attention modules.

### 6.1 LoRA Adapter Formulation
For a weight matrix projection $W_0 \in \mathbb{R}^{d \times k}$, we decompose the weight update $\Delta W$ into two low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ where rank $r \ll \min(d, k)$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B A)$$

We place LoRA adapters on the query, key, value, and output projection layers of the self-attention and cross-attention blocks.

*   **Rank ($r$)**: 16
*   **Alpha ($\alpha$)**: 32
*   **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `out_proj`

### 6.2 Optimization & Regularization Strategy
- **Optimizer**: AdamW with weight decay of 0.01.
- **Precision**: Float16 mixed precision to optimize VRAM footprint.
- **Gradient Accumulation**: Set to 4 steps to simulate larger batch training on modest GPUs.

---

## 7. Data Engineering Pipeline

A robust low-resource data pipeline is necessary to maximize training quality.

```
       [Raw Corpora (600K Pairs)]
                   |
        [Unicode Normalization] (NFC normalization)
                   |
        [Script Normalization] (Devanagari to Latin for Gondi)
                   |
       [Fuzzy Deduplication] (MinHash LSH index check)
                   |
    [Language Identification Filtering] (FastText LID classifier)
                   |
          +--------+--------+
          |                 |
          v                 v
   [Parallel Data]     [Monolingual Gondi / Hindi]
          |                 |
          |                 v
          |         [Backtranslation Loop]
          |         (Generate synthetic pairs)
          |                 |
          +--------+--------+
                   |
                   v
      [Final Preprocessed Dataset]
```

### 7.1 Denoising & Augmentation
- **Subword Regularization**: Apply BPE-dropout ($p=0.1$) during training batch generation to create alternate segmentations, preventing the model from memorizing fixed subword chunks.
- **Backtranslation**: Standard monolingual Gondi sentences are translated back into Hindi using a preliminary model, appending the resulting pairs as synthetic parallel training inputs.

---

## 8. Robust Decoding: Minimum Bayes Risk (MBR)

Standard Beam Search is highly susceptible to hallucinations in low-resource regimes, often outputting repetitive phrases or switching into dominant languages (like Hindi or English). 

To counter this, our system employs **Minimum Bayes Risk (MBR) Decoding**.

### 8.1 Core Mechanism
1. **Candidate Pool Generation**: We sample $K = 16$ candidate translations from the decoder using temperature-scaled ancestral sampling ($T = 0.65$).
2. **Pairwise Utility Scoring**: We compute a similarity score (e.g., $chrF$ or $BLEU$) between all candidate pairs in the pool.
3. **Consensus Selection**: We choose the candidate that minimizes the expected risk (maximizes utility) relative to all other candidates:

$$\hat{Y} = \arg\max_{Y \in \mathcal{U}} \frac{1}{K} \sum_{j=1}^K \text{Utility}(Y, Y^{(j)})$$

where $\mathcal{U}$ is the candidate translation pool, and $\text{Utility}$ is the $chrF$ string-matching metric. This eliminates statistical outliers and hallucinations.

---

## 9. Architectural Comparisons

| Feature | Standard T5 | ByT5 | Vanilla NLLB | Our Custom Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **Tokenizer Strategy** | WordPiece | Character-level | SentencePiece (256k) | Custom BPE (32k) with language tags |
| **Morphology Handling** | Moderate | Good (slow) | Moderate | Excellent (Gated-GeLU FFN + BPE) |
| **Low-Resource Tuning** | Poor | Poor (overfits) | Heavy (overfits) | Designed for ~600K lines |
| **Encoder-Decoder Layout**| Symmetric | Asymmetric | Symmetric | Asymmetric (8 Enc / 4 Dec layers) |
| **Positional Encoding** | Relative | Relative | Absolute | Relative Position Bias (T5) |
| **Retrieval Memory** | None | None | None | Integrated Translation Memory fusion |
| **Parameter Efficiency** | Full fine-tune | Full fine-tune | Full fine-tune | LoRA Adapter selective tuning |
| **Inference Cost** | Low | High (char step) | High (vocab size) | Ultra-low (asymmetric + small vocab) |

---

## 10. Research Novelty, Deployment, and Future Roadmap

### 10.1 Novelty Summary
This architecture provides a scalable framework for low-resource tribal NLP by addressing language barriers without requiring massive high-compute infrastructures. The combination of **asymmetric layer sizing**, **Gated-GeLU activation routing**, and **hybrid retrieval-augmented fusion** mitigates data sparsity while preventing target hallucinations.

### 10.2 Production Deployment Layout
*   **Backend Serving**: PyTorch model converted to ONNX/TensorRT format.
*   **Retrieval Service**: A lightweight FAISS vector database running side-by-side with the translation engine.
*   **Web API**: Fast API service hosting the translation pipeline:

```
[User Request] -> [API Handler] -> [FAISS Retrieval Search] -> [Model Inference (MBR)] -> [Post-Process] -> [Response]
```

### 10.3 Future Roadmap
1. **Extensibility to Bhili/Santali**: Expanding the SentencePiece vocabulary with language tags `__bhi_Deva__` and `__sat_Olck__` (Ol Chiki script).
2. **Speech Translation Integration**: Integrating a Whisper-based encoder ahead of the asymmetric translation decoder for direct speech-to-text translation of tribal dialects.
