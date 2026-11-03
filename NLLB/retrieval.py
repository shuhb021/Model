import math
import collections
import re
from typing import List, Tuple, Dict, Optional

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fuzzy_similarity(s1: str, s2: str) -> float:
    """
    Normalizes Levenshtein distance to a similarity score between 0.0 and 1.0.
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF SEMANTIC SEARCH ENGINE (zero dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFSearcher:
    """
    A pure-python TF-IDF semantic vector searcher.
    Avoids external dependencies (like scikit-learn or FAISS) for robust execution,
    while matching performance on sentence-level retrieval tasks.
    """
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.num_docs = len(corpus)
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[int, float]] = []
        self.doc_lengths: List[float] = []

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and tokenise words
        return re.findall(r"\w+", text.lower())

    def _build_index(self):
        doc_tfs = []
        df = collections.Counter()
        
        # 1. Compute term frequencies and document frequencies
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            tf = collections.Counter(tokens)
            doc_tfs.append(tf)
            for token in tf.keys():
                df[token] += 1
                
        # 2. Build vocabulary
        all_words = sorted(list(df.keys()))
        self.vocab = {word: i for i, word in enumerate(all_words)}
        
        # 3. Compute IDF
        for word, count in df.items():
            # Standard IDF formula with smoothing
            self.idf[word] = math.log((self.num_docs + 1) / (count + 0.5)) + 1.0

        # 4. Compute TF-IDF document vectors
        for tf in doc_tfs:
            vector = {}
            length_sq = 0.0
            for word, count in tf.items():
                if word in self.vocab:
                    word_idx = self.vocab[word]
                    tfidf_val = count * self.idf[word]
                    vector[word_idx] = tfidf_val
                    length_sq += tfidf_val ** 2
            self.doc_vectors.append(vector)
            self.doc_lengths.append(math.sqrt(length_sq))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Executes a cosine similarity search between the query and index.
        Returns: List of tuples (document_index, cosine_similarity_score)
        """
        tokens = self._tokenize(query)
        tf = collections.Counter(tokens)
        
        # Compute query vector
        query_vector = {}
        query_len_sq = 0.0
        for word, count in tf.items():
            if word in self.vocab:
                word_idx = self.vocab[word]
                tfidf_val = count * self.idf[word]
                query_vector[word_idx] = tfidf_val
                query_len_sq += tfidf_val ** 2
        
        query_len = math.sqrt(query_len_sq)
        if query_len == 0:
            return []

        results = []
        for doc_idx, doc_vec in enumerate(self.doc_vectors):
            doc_len = self.doc_lengths[doc_idx]
            if doc_len == 0:
                continue
            
            # Compute dot product
            dot_product = 0.0
            for word_idx, query_val in query_vector.items():
                if word_idx in doc_vec:
                    dot_product += query_val * doc_vec[word_idx]
            
            similarity = dot_product / (query_len * doc_len)
            results.append((doc_idx, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ─────────────────────────────────────────────────────────────────────────────
# FUSION TRANSLATION MEMORY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class TranslationMemory:
    """
    Combines fuzzy search and semantic search to return the most relevant
    translation match, formatting it into a prefix context for the MT model.
    """
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str]):
        assert len(src_sentences) == len(tgt_sentences), "Source and Target corpora must be aligned"
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.semantic_searcher = TFIDFSearcher(src_sentences)

    def retrieve(self, query_src: str, threshold: float = 0.6) -> Optional[Tuple[str, str, float]]:
        """
        Retrieves the best translation memory pair from the corpus.
        Combines Semantic search (for filtering) with Levenshtein fuzzy score (for accuracy).
        """
        # 1. Filter candidates using semantic TF-IDF (Top-15)
        candidates = self.semantic_searcher.search(query_src, top_k=15)
        
        best_match = None
        best_score = -1.0
        
        # 2. Rescore candidates using fine-grained fuzzy similarity
        for doc_idx, _ in candidates:
            candidate_src = self.src_sentences[doc_idx]
            candidate_tgt = self.tgt_sentences[doc_idx]
            
            score = fuzzy_similarity(query_src, candidate_src)
            if score > best_score:
                best_score = score
                best_match = (candidate_src, candidate_tgt, score)

        if best_match and best_score >= threshold:
            return best_match
        return None

    def fuse_context(self, query_src: str, threshold: float = 0.6) -> str:
        """
        Fuses the translation memory match into the input sequence.
        Format: [TM_START] source_match [TM_SEP] target_match [TM_END] [LANG_TAG] query_src
        """
        match = self.retrieve(query_src, threshold=threshold)
        if match:
            src_match, tgt_match, score = match
            return f"[TM_START] {src_match} [TM_SEP] {tgt_match} [TM_END] {query_src}"
        return query_src

if __name__ == "__main__":
    import sys
    # Avoid print encoding crashes on Windows PowerShell
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    # Demo corpus
    src_db = [
        "aggator vat itteke nahan",
        "an akkal adenlasi pator",
        "gondi pargoti adenla manval",
        "vatusval akkurpok"
    ]
    tgt_db = [
        "वह आदमी वहां गया था",
        "उसने बुद्धि से काम किया",
        "गोंडी भाषा बहुत पुरानी है",
        "धूप में सूखा हुआ फल"
    ]
    
    tm = TranslationMemory(src_db, tgt_db)
    
    # Test retrieve
    test_query = "aggator vat itteke nahan"
    res = tm.retrieve(test_query)
    print("Exact Query Match:")
    print(f"Query: {test_query} -> Found: {res}")
    
    # Test fuzzy retrieve
    fuzzy_query = "aggator vat itteke"
    res_fuzzy = tm.retrieve(fuzzy_query)
    print("\nFuzzy Query Match:")
    print(f"Query: {fuzzy_query} -> Found: {res_fuzzy}")
    
    # Test fusion formatting
    fused = tm.fuse_context(fuzzy_query)
    print("\nFused Prefix Output:")
    print(fused)
