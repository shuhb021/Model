"""
Retrieval-Augmented Translation Memory (TM)
────────────────────────────────────────────
Dual retrieval pipeline per spec:

    [Query Source Sentence]
           ├── Fuzzy Matching  (Levenshtein distance)
           └── Dense Retrieval (Sentence-Transformer + FAISS)
                     ↓
            [Reranking: Length + Cosine Similarity]
                     ↓
            [Fusion Context Generation]
         "TM: {Src} = {Tgt} <SEP> {Query}"
                     ↓
            [Decoder Generation Target]

Fusion rule:
    similarity > 0.85 → copy matching structure
    similarity < 0.85 → translate natively (no TM prefix)
"""

import math
from typing import List, Optional, Tuple


# ─── Levenshtein Distance (Fuzzy Matching) ────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein implementation."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Normalized similarity in [0, 1].
    similarity = 1 - edit_distance / max_length
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


# ─── TF-IDF Semantic Similarity (lightweight fallback) ───────────────────────

def _term_freq(tokens: List[str]) -> dict:
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def tfidf_cosine_similarity(s1: str, s2: str) -> float:
    """
    Approximate cosine similarity using TF-IDF bag-of-words.
    For production, replace with a real sentence-transformer + FAISS index.
    """
    t1 = s1.lower().split()
    t2 = s2.lower().split()
    tf1 = _term_freq(t1)
    tf2 = _term_freq(t2)

    vocab = set(tf1.keys()) | set(tf2.keys())
    v1 = [tf1.get(w, 0) for w in vocab]
    v2 = [tf2.get(w, 0) for w in vocab]

    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a**2 for a in v1))
    norm2 = math.sqrt(sum(b**2 for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ─── Translation Memory Index ─────────────────────────────────────────────────

class TranslationMemory:
    """
    In-memory translation memory with dual retrieval:
    - Fuzzy (Levenshtein) for character-level similarity
    - Semantic (TF-IDF cosine / sentence-transformer) for meaning

    For large corpora (600K pairs), replace the linear scan with FAISS:
        pip install faiss-cpu sentence-transformers
    """

    def __init__(self, fusion_threshold: float = 0.85):
        self.entries: List[Tuple[str, str]] = []   # (src, tgt) pairs
        self.fusion_threshold = fusion_threshold

    def add(self, src: str, tgt: str):
        self.entries.append((src.strip(), tgt.strip()))

    def add_bulk(self, pairs: List[Tuple[str, str]]):
        for src, tgt in pairs:
            self.add(src, tgt)
        print(f"[OK] TM index built with {len(self.entries):,} entries")

    def _rerank_score(self, query: str, candidate_src: str) -> float:
        """Combined fuzzy + semantic score with length penalty."""
        fuzzy_sim    = normalized_levenshtein_similarity(query, candidate_src)
        semantic_sim = tfidf_cosine_similarity(query, candidate_src)

        # Length penalty: penalise very different length ratios
        q_len = max(len(query.split()), 1)
        c_len = max(len(candidate_src.split()), 1)
        len_ratio = min(q_len, c_len) / max(q_len, c_len)

        return 0.5 * fuzzy_sim + 0.4 * semantic_sim + 0.1 * len_ratio

    def retrieve(self, query: str, top_k: int = 3,
                 window: int = 5_000) -> List[Tuple[float, str, str]]:
        """
        Retrieve top-k matches for a query.
        For large indices, `window` limits linear scan to last N entries.

        Returns:
            List of (score, src, tgt) sorted by score descending
        """
        candidates = self.entries[-window:] if len(self.entries) > window else self.entries

        scored = [
            (self._rerank_score(query, src), src, tgt)
            for src, tgt in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def get_fusion_prefix(self, query: str,
                          tgt_lang_tag: str,
                          tm_start: str = "[TM_START]",
                          tm_sep: str   = "[TM_SEP]",
                          tm_end: str   = "[TM_END]") -> Optional[str]:
        """
        If best match similarity > threshold, return TM-augmented prefix.
        Otherwise return None (translate natively).

        Prefix format:
            [TM_START] {tm_src} [TM_SEP] {tm_tgt} [TM_END] {tgt_lang_tag} {query}
        """
        if not self.entries:
            return None

        results = self.retrieve(query, top_k=1)
        if not results:
            return None

        best_score, best_src, best_tgt = results[0]

        if best_score >= self.fusion_threshold:
            return (f"{tm_start} {best_src} "
                    f"{tm_sep} {best_tgt} "
                    f"{tm_end} {tgt_lang_tag} {query}")

        return None   # low similarity → native translation


# ─── FAISS Integration (production stub) ─────────────────────────────────────

class FAISSTranslationMemory:
    """
    Production-grade dense retrieval using FAISS + sentence-transformers.
    Requires: pip install faiss-cpu sentence-transformers
    
    Drop-in replacement for TranslationMemory for the 600K corpus.
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE",
                 fusion_threshold: float = 0.85):
        """
        LaBSE (Language-agnostic BERT Sentence Embeddings) is recommended
        for multilingual retrieval covering Gondi/Hindi/English.
        """
        self.fusion_threshold = fusion_threshold
        self.model_name       = model_name
        self._index           = None
        self._tgts: List[str] = []
        self._srcs: List[str] = []

        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.encoder = SentenceTransformer(model_name)
            self.faiss   = faiss
            print(f"[OK] FAISS+SentenceTransformer ready | model={model_name}")
        except ImportError:
            self.encoder = None
            print("[WARN] Install with: pip install faiss-cpu sentence-transformers")

    def build_index(self, pairs: List[Tuple[str, str]], batch_size: int = 256):
        """Encode all source sentences and build FAISS flat IP index."""
        if self.encoder is None:
            raise RuntimeError("sentence-transformers not installed")

        import numpy as np

        srcs = [s for s, _ in pairs]
        tgts = [t for _, t in pairs]

        print(f"  Encoding {len(srcs):,} sentences for FAISS index...")
        embeddings = self.encoder.encode(
            srcs, batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self._index = self.faiss.IndexFlatIP(dim)   # cosine similarity (normalized)
        self._index.add(embeddings.astype(np.float32))
        self._srcs = srcs
        self._tgts = tgts
        print(f"[OK] FAISS index ready | {self._index.ntotal:,} vectors | dim={dim}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, str, str]]:
        if self._index is None:
            raise RuntimeError("Call build_index() first")
        import numpy as np

        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(q_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), self._srcs[idx], self._tgts[idx]))
        return results

    def get_fusion_prefix(self, query: str, tgt_lang_tag: str,
                          tm_start: str = "[TM_START]",
                          tm_sep: str   = "[TM_SEP]",
                          tm_end: str   = "[TM_END]") -> Optional[str]:
        results = self.retrieve(query, top_k=1)
        if not results:
            return None
        best_score, best_src, best_tgt = results[0]
        if best_score >= self.fusion_threshold:
            return (f"{tm_start} {best_src} "
                    f"{tm_sep} {best_tgt} "
                    f"{tm_end} {tgt_lang_tag} {query}")
        return None


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tm = TranslationMemory(fusion_threshold=0.85)
    tm.add_bulk([
        ("I am going home",    "main ghar jata hoon"),
        ("She is reading",     "woh padh rahi hai"),
        ("The tree is tall",   "ped uncha hai"),
        ("Water is cold",      "paani thanda hai"),
    ])

    query = "I am going to home"
    prefix = tm.get_fusion_prefix(query, tgt_lang_tag="__hin_Deva__")

    if prefix:
        print(f"[OK] TM Prefix generated:\n   {prefix}")
    else:
        print("  -> Native translation (low similarity)")

    results = tm.retrieve(query, top_k=3)
    print("\nTop-3 TM hits:")
    for score, src, tgt in results:
        print(f"  {score:.3f} | {src} -> {tgt}")
