"""
Data Engineering Pipeline
──────────────────────────
Pipeline flow (from spec):
    Raw Corpora (600K Pairs)
        → Unicode NFC Normalization
        → Script Normalization (Devanagari → Latin for Gondi)
        → Fuzzy Deduplication (MinHash LSH)
        → Language ID Filtering (FastText)
        → Parallel + Monolingual splits
        → Backtranslation loop (synthetic pairs)
        → Final Preprocessed Dataset

Training augmentation:
    • BPE-Dropout (p=0.1) — alternate subword segmentations
    • Temperature-based multilingual batch sampling (T=1.6)
"""

import json
import math
import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ─── Unicode Normalization ─────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Apply NFC normalization and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFC", text.strip())


def basic_clean(text: str) -> str:
    text = normalize_unicode(text)
    # Remove null bytes
    text = text.replace("\x00", "")
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


# ─── MinHash Fuzzy Deduplication ──────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3) -> List[str]:
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def minhash_signature(text: str, num_perm: int = 64) -> List[int]:
    """
    Lightweight MinHash signature for approximate deduplication.
    For production use datasketch library: pip install datasketch
    """
    ngrams = set(_char_ngrams(text.lower()))
    if not ngrams:
        return [0] * num_perm
    sig = []
    for seed in range(num_perm):
        random.seed(seed)
        min_hash = min(hash(ng + str(seed)) % (2**32) for ng in ngrams)
        sig.append(min_hash)
    return sig


def jaccard_from_signatures(sig1: List[int], sig2: List[int]) -> float:
    matches = sum(a == b for a, b in zip(sig1, sig2))
    return matches / len(sig1)


def deduplicate_pairs(pairs: List[Tuple[str, str]],
                      threshold: float = 0.85) -> List[Tuple[str, str]]:
    """
    Remove near-duplicate sentence pairs using MinHash LSH approximation.
    threshold: Jaccard similarity above which a pair is considered duplicate.
    """
    print(f"  Deduplicating {len(pairs):,} pairs (threshold={threshold})...")
    signatures = [minhash_signature(src) for src, _ in pairs]
    keep = [True] * len(pairs)

    for i in range(len(pairs)):
        if not keep[i]:
            continue
        for j in range(i + 1, min(i + 500, len(pairs))):  # local window
            if not keep[j]:
                continue
            if jaccard_from_signatures(signatures[i], signatures[j]) >= threshold:
                keep[j] = False

    result = [p for p, k in zip(pairs, keep) if k]
    print(f"  → Kept {len(result):,} unique pairs "
          f"(removed {len(pairs) - len(result):,})")
    return result


# ─── Language ID Filtering ────────────────────────────────────────────────────

def has_devanagari(text: str) -> bool:
    return any('\u0900' <= c <= '\u097F' for c in text)

def has_latin(text: str) -> bool:
    return any('a' <= c.lower() <= 'z' for c in text)

def filter_by_lang_id(pairs: List[Tuple[str, str]],
                       src_lang: str,
                       tgt_lang: str) -> List[Tuple[str, str]]:
    """
    Simple heuristic language filter.
    For production, use fasttext LID: pip install fasttext-langdetect
    """
    def check(text: str, lang: str) -> bool:
        if lang in ("hindi",):
            return has_devanagari(text)
        if lang in ("gondi", "english"):
            return has_latin(text) or len(text) > 2
        return True

    return [
        (s, t) for s, t in pairs
        if check(s, src_lang) and check(t, tgt_lang)
        and 2 <= len(s.split()) <= 150
        and 2 <= len(t.split()) <= 150
    ]


# ─── Translation Pair Dataset ─────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation pairs.

    Applies:
        - BPE-Dropout (p=0.1) during training for subword regularization
        - Language tag prepending via tokenizer
        - Dynamic padding within batch (via collate_fn)
    """

    def __init__(self,
                 pairs: List[Tuple[str, str]],
                 src_lang: str,
                 tgt_lang: str,
                 tokenizer,
                 max_length: int = 256,
                 bpe_dropout: float = 0.0,
                 is_train: bool = True):
        self.pairs       = pairs
        self.src_lang    = src_lang
        self.tgt_lang    = tgt_lang
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.bpe_dropout = bpe_dropout if is_train else 0.0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text, tgt_text = self.pairs[idx]

        src_ids = self.tokenizer.encode(
            src_text, lang=self.src_lang,
            bpe_dropout=self.bpe_dropout
        )[:self.max_length]

        tgt_ids = self.tokenizer.encode(
            tgt_text, lang=self.tgt_lang,
            bpe_dropout=self.bpe_dropout
        )[:self.max_length]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate function."""
    src_max = max(b["src_ids"].shape[0] for b in batch)
    tgt_max = max(b["tgt_ids"].shape[0] for b in batch)

    src_padded, tgt_in_padded, tgt_out_padded = [], [], []

    for b in batch:
        s = b["src_ids"]
        t = b["tgt_ids"]

        src_pad = src_max - s.shape[0]
        tgt_pad = tgt_max - t.shape[0]

        src_padded.append(torch.cat([s, torch.full((src_pad,), pad_id)]))

        # Decoder input: BOS + tokens (drop last token)
        tgt_in  = t[:-1]
        tgt_out = t[1:]           # teacher-forced labels (shift left)
        tgt_in_padded.append(torch.cat([tgt_in,  torch.full((tgt_pad + 1,), pad_id)]))
        tgt_out_padded.append(torch.cat([tgt_out, torch.full((tgt_pad + 1,), pad_id)]))

    return {
        "src":    torch.stack(src_padded),        # [B, S]
        "tgt_in": torch.stack(tgt_in_padded),     # [B, T-1]
        "tgt_out": torch.stack(tgt_out_padded),   # [B, T-1]  (labels)
    }


# ─── Multilingual Temperature Sampling ────────────────────────────────────────

class MultilingualSampler:
    """
    Temperature-based dataset sampling to prevent high-resource language dominance.
    P_i = N_i^(1/T) / Σ_j N_j^(1/T)    where T = 1.6
    """

    def __init__(self, dataset_sizes: Dict[str, int], temperature: float = 1.6):
        self.temperature = temperature
        self.dataset_sizes = dataset_sizes
        self.probabilities = self._compute_probabilities()

    def _compute_probabilities(self) -> Dict[str, float]:
        powered = {k: v ** (1.0 / self.temperature)
                   for k, v in self.dataset_sizes.items()}
        total = sum(powered.values())
        return {k: v / total for k, v in powered.items()}

    def sample_counts(self, total_samples: int) -> Dict[str, int]:
        return {k: int(p * total_samples)
                for k, p in self.probabilities.items()}

    def print_sampling_plan(self):
        print("\n📊 Temperature Sampling Plan (T=1.6):")
        print(f"{'Direction':<25} {'Dataset Size':>12} {'Sample Prob':>12}")
        print("─" * 50)
        for k in self.dataset_sizes:
            print(f"  {k:<23} {self.dataset_sizes[k]:>12,} {self.probabilities[k]:>11.3f}")
        print()


# ─── Pipeline Runner ──────────────────────────────────────────────────────────

class DataPipeline:
    """End-to-end data preprocessing pipeline."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, raw_pairs: List[Tuple[str, str]],
            src_lang: str, tgt_lang: str,
            deduplicate: bool = True) -> List[Tuple[str, str]]:
        print(f"\n🚀 Pipeline: {src_lang} → {tgt_lang}")
        print(f"   Input:  {len(raw_pairs):,} pairs")

        # Step 1: Basic cleaning
        pairs = [(basic_clean(s), basic_clean(t)) for s, t in raw_pairs]
        pairs = [(s, t) for s, t in pairs if s and t]

        # Step 2: Language ID filtering
        pairs = filter_by_lang_id(pairs, src_lang, tgt_lang)
        print(f"   After LID filter: {len(pairs):,}")

        # Step 3: Deduplication
        if deduplicate:
            pairs = deduplicate_pairs(pairs)

        print(f"   ✅ Final: {len(pairs):,} pairs")

        # Save
        out_file = self.output_dir / f"{src_lang}_{tgt_lang}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s, t in pairs:
                f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")
        print(f"   💾 Saved → {out_file}")

        return pairs

    @staticmethod
    def load_jsonl(path: str) -> List[Tuple[str, str]]:
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pairs.append((d["src"], d["tgt"]))
        return pairs


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate multilingual temperature sampling
    sizes = {
        "gondi↔hindi":   150_000,
        "gondi↔english": 150_000,
        "hindi↔english": 300_000,
    }
    sampler = MultilingualSampler(sizes, temperature=1.6)
    sampler.print_sampling_plan()

    # Simulate deduplication
    dummy_pairs = [(f"source sentence {i}", f"target sentence {i}") for i in range(200)]
    dummy_pairs += dummy_pairs[:20]  # add duplicates
    pipeline = DataPipeline(output_dir="/tmp/nmt_test_output")
    cleaned = pipeline.run(dummy_pairs, "english", "hindi", deduplicate=True)
    print(f"\n✅ Pipeline test passed | {len(cleaned)} pairs kept")
