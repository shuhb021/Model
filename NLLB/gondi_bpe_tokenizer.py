"""
GondiTokenizer — A from-scratch Byte-Pair Encoding (BPE) Tokenizer
====================================================================
Language  : Gondi (low-resource, highly agglutinative, Latin script)
Libraries : Python Standard Library only  (csv, re, json, collections)

BPE Algorithm recap
-------------------
1. Start with a character-level vocabulary.
2. Represent every word as a sequence of character-tokens separated by a
   special end-of-word marker (Ġ, shown here as "</w>").
3. Repeatedly find the most-frequent adjacent pair, merge it into one new
   token, and record the merge rule.
4. Stop when the desired vocab_size is reached.
5. At inference time replay the recorded merge rules in order to encode
   any new word; decode by reversing the process.
"""

import csv
import re
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_gondi_text(text: str) -> str:
    """
    Clean a raw Gondi token/phrase from the CSV.

    Steps
    -----
    • Strip leading/trailing whitespace.
    • Remove content inside parentheses  e.g. "akurpok (mahna)" → "akurpok"
    • Remove content inside square brackets.
    • Collapse multiple internal spaces to one.
    • Lowercase everything (the CSV is already Latin-script lowercase, but
      let's be safe).
    • Return only the *first* token if multiple words remain after cleaning
      (multi-word entries are not useful for a single-word tokenizer).
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    # Remove BOM / non-printable leading chars
    text = text.lstrip("\ufeff\u200b")
    # Drop parenthesised / bracketed material
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase
    text = text.lower()
    # Keep only the first token (handles "an bav", "als puls", etc.)
    text = text.split()[0] if text.split() else ""
    # Keep letters, hyphens, and Devanagari characters
    text = re.sub(r"[^a-z\-\u0900-\u097F]", "", text)
    return text


def load_gondi_corpus(csv_path: str) -> List[str]:
    """
    Load the Gondi column from the CSV and return a clean list of words.
    """
    words = []
    with open(csv_path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("Gondi", "")
            cleaned = preprocess_gondi_text(raw)
            if cleaned:
                words.append(cleaned)
    return words


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BPE CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

END_OF_WORD = "</w>"   # marks word boundary; survives the entire pipeline


def _word_to_token_seq(word: str) -> Tuple[str, ...]:
    """
    Split a word into its initial character-level BPE representation.

    "aggator" → ('a', 'g', 'g', 'a', 't', 'o', 'r</w>')

    The last character is fused with END_OF_WORD so the tokenizer always
    knows where a word ends.  This also lets it learn suffix merges such as
    'or</w>' → 'ator</w>' naturally.
    """
    if not word:
        return ()
    chars = list(word)
    chars[-1] = chars[-1] + END_OF_WORD
    return tuple(chars)


def _build_vocab(corpus: List[str]) -> Dict[Tuple[str, ...], int]:
    """
    Build a frequency dictionary  { token_sequence: count }.

    Multiple occurrences of the same word are counted once per occurrence
    (the dictionary value holds the total frequency).
    """
    vocab: Dict[Tuple[str, ...], int] = defaultdict(int)
    for word in corpus:
        seq = _word_to_token_seq(word)
        if seq:
            vocab[seq] += 1
    return dict(vocab)


def _get_pair_frequencies(
    vocab: Dict[Tuple[str, ...], int]
) -> Counter:
    """
    Count every adjacent pair across all word sequences, weighted by frequency.

    For  vocab = { ('a','g','g','ator</w>'): 3 }
    pairs counted: ('a','g')×3, ('g','g')×3, ('g','ator</w>')×3
    """
    pair_freq: Counter = Counter()
    for token_seq, freq in vocab.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_freq[pair] += freq
    return pair_freq


def _merge_pair(
    vocab: Dict[Tuple[str, ...], int],
    best_pair: Tuple[str, str],
) -> Dict[Tuple[str, ...], int]:
    """
    Apply one BPE merge rule to the entire vocabulary.

    Replaces every adjacent occurrence of best_pair with the concatenated token.
    Returns a NEW vocabulary dict (original is untouched).
    """
    a, b = best_pair
    merged_token = a + b
    new_vocab: Dict[Tuple[str, ...], int] = {}

    for token_seq, freq in vocab.items():
        new_seq: List[str] = []
        i = 0
        while i < len(token_seq):
            # Try to match the pair at position i
            if i < len(token_seq) - 1 and token_seq[i] == a and token_seq[i + 1] == b:
                new_seq.append(merged_token)
                i += 2          # skip both tokens
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_vocab[tuple(new_seq)] = freq

    return new_vocab


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TOKENIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class GondiTokenizer:
    """
    Byte-Pair Encoding tokeniser tuned for Gondi (Latin script).

    Public API
    ----------
    train(corpus, vocab_size)   – learn BPE merges from a word list
    encode(text)                – raw string  →  list[int]   (token IDs)
    decode(ids)                 – list[int]   →  string
    save(path)                  – persist vocab + merges to JSON
    load(path)  [classmethod]   – restore from JSON
    """

    # Special tokens
    PAD_TOKEN   = "[PAD]"
    UNK_TOKEN   = "[UNK]"
    START_TOKEN = "[START]"
    END_TOKEN   = "[END]"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

    def __init__(self) -> None:
        # token  → id
        self.token_to_id: Dict[str, int] = {}
        # id     → token
        self.id_to_token: Dict[int, str] = {}
        # ordered list of merge rules  [(a, b), ...]
        self.merges: List[Tuple[str, str]] = []
        # set for O(1) lookup during encoding
        self._merge_set: Dict[Tuple[str, str], int] = {}

        self._trained = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    # ── training ─────────────────────────────────────────────────────────────

    def train(
        self,
        corpus: List[str],
        vocab_size: int = 300,
        verbose: bool = True,
    ) -> None:
        """
        Learn BPE merges from a list of Gondi words.

        Parameters
        ----------
        corpus     : list of clean (preprocessed) Gondi words
        vocab_size : target vocabulary size (including special tokens and
                     the initial character alphabet)
        verbose    : print progress every 50 merges
        """
        if verbose:
            print(f"[GondiTokenizer] Training on {len(corpus)} words …")

        # ── Step 0: seed special tokens ──────────────────────────────────────
        for tok in self.SPECIAL_TOKENS:
            self._add_token(tok)

        # ── Step 1: build initial character vocabulary ────────────────────────
        # Collect every character that appears in the corpus, plus END_OF_WORD
        # fused variants  (e.g. 'r</w>', 'l</w>').
        char_vocab: set = set()
        for word in corpus:
            if not word:
                continue
            seq = _word_to_token_seq(word)
            char_vocab.update(seq)

        # Sort for determinism; digits/punctuation first, then alpha
        for ch in sorted(char_vocab):
            self._add_token(ch)

        if verbose:
            print(
                f"[GondiTokenizer] Base alphabet: {len(char_vocab)} tokens  "
                f"(total so far: {len(self.token_to_id)})"
            )

        # ── Step 2: BPE merge loop ────────────────────────────────────────────
        current_vocab = _build_vocab(corpus)
        num_merges_needed = vocab_size - len(self.token_to_id)

        if num_merges_needed <= 0:
            if verbose:
                print("[GondiTokenizer] vocab_size already reached by base alphabet.")
        else:
            for merge_idx in range(num_merges_needed):
                pair_freq = _get_pair_frequencies(current_vocab)
                if not pair_freq:
                    if verbose:
                        print("[GondiTokenizer] No more pairs — stopping early.")
                    break

                best_pair = pair_freq.most_common(1)[0][0]
                new_token = best_pair[0] + best_pair[1]

                # Record merge rule
                self.merges.append(best_pair)
                self._merge_set[best_pair] = merge_idx

                # Add merged token to vocab
                self._add_token(new_token)

                # Apply the merge
                current_vocab = _merge_pair(current_vocab, best_pair)

                if verbose and (merge_idx + 1) % 50 == 0:
                    print(
                        f"  merge {merge_idx + 1:>4}/{num_merges_needed}  "
                        f"'{best_pair[0]}' + '{best_pair[1]}' → '{new_token}'  "
                        f"(freq={pair_freq[best_pair]})"
                    )

        self._trained = True
        if verbose:
            print(
                f"[GondiTokenizer] Training complete.  "
                f"Final vocabulary size: {len(self.token_to_id)}"
            )

    # ── encoding ─────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a single (clean) word.

        Returns a list of sub-word tokens (the last one ends with </w>).
        Unknown characters are replaced with [UNK].
        """
        seq = list(_word_to_token_seq(word))
        # Replace any character not in vocabulary with UNK
        seq = [
            ch if ch in self.token_to_id else self.UNK_TOKEN
            for ch in seq
        ]

        if len(seq) < 2:
            return seq

        # Replay merges in training order (greedy left-to-right per merge rule)
        # For efficiency: iterate over recorded merges and apply each once
        # (standard BPE encoding strategy)
        for pair in self.merges:
            a, b = pair
            i = 0
            new_seq: List[str] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(a + b)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
            if len(seq) == 1:
                break           # fully merged; nothing more to do

        return seq

    def encode(self, text: str) -> List[int]:
        """
        Convert a raw Gondi string to a list of token IDs.

        Multi-word input is supported: each word is tokenised independently.
        """
        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained yet. Call .train() first.")

        text = preprocess_gondi_text(text)
        # Re-allow spaces so the user can pass short phrases
        words = text.split() if text else []
        unk_id = self.token_to_id[self.UNK_TOKEN]

        ids: List[int] = []
        for word in words:
            tokens = self._tokenize_word(word)
            for tok in tokens:
                ids.append(self.token_to_id.get(tok, unk_id))
        return ids

    # ── decoding ─────────────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a human-readable Gondi string.

        Sub-words that end with </w> signal the end of a word; a space is
        inserted after them (except at the very end of the sequence).
        """
        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained yet. Call .train() first.")

        tokens = [self.id_to_token.get(i, self.UNK_TOKEN) for i in ids]
        result_parts: List[str] = []
        word_buf: List[str] = []

        for tok in tokens:
            if tok in self.SPECIAL_TOKENS:
                if word_buf:
                    result_parts.append("".join(word_buf))
                    word_buf = []
                # skip special tokens in output
                continue

            if tok.endswith(END_OF_WORD):
                word_buf.append(tok[: -len(END_OF_WORD)])   # strip </w>
                result_parts.append("".join(word_buf))
                word_buf = []
            else:
                word_buf.append(tok)

        # Flush any remaining buffer (shouldn't happen with well-formed data)
        if word_buf:
            result_parts.append("".join(word_buf))

        return " ".join(result_parts)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the learned vocabulary and merge rules to a JSON file.

        Schema
        ------
        {
          "token_to_id": { "<token>": <int>, … },
          "merges":       [ ["a", "b"], … ]
        }
        """
        data = {
            "token_to_id": self.token_to_id,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"[GondiTokenizer] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "GondiTokenizer":
        """
        Restore a previously saved GondiTokenizer from a JSON file.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        tokenizer = cls()
        tokenizer.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._merge_set = {tuple(p): i for i, p in enumerate(tokenizer.merges)}
        tokenizer._trained = True

        print(
            f"[GondiTokenizer] Model loaded ← {path}  "
            f"(vocab={len(tokenizer.token_to_id)}, merges={len(tokenizer.merges)})"
        )
        return tokenizer

    # ── convenience ──────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """Return the token *strings* (useful for debugging)."""
        text = preprocess_gondi_text(text)
        words = text.split() if text else []
        tokens: List[str] = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        return tokens

    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __repr__(self) -> str:
        status = f"vocab={self.vocab_size()}, merges={len(self.merges)}" if self._trained else "untrained"
        return f"GondiTokenizer({status})"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN — DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> None:
    CSV_PATH   = "DS1 WB.csv"
    MODEL_PATH = "gondi_bpe_model.json"
    VOCAB_SIZE = 400   # Adjusted for the dataset size

    # ── 4.1  Load & inspect corpus ───────────────────────────────────────────
    _print_section("STEP 1 — Load Corpus")
    corpus = load_gondi_corpus(CSV_PATH)
    print(f"  Total words after cleaning : {len(corpus)}")
    print(f"  Unique words               : {len(set(corpus))}")
    print(f"  Sample (first 20)          : {corpus[:20]}")

    # Quick peek at Gondi-specific phonotactics
    double_consonants = [w for w in corpus if re.search(r"(.)\1", w)]
    suffix_tor   = [w for w in corpus if w.endswith("tor")]
    suffix_val   = [w for w in corpus if w.endswith("val")]
    suffix_lasi  = [w for w in corpus if w.endswith("lasi")]

    print(f"\n  Words with double consonants : {len(double_consonants)}  e.g. {double_consonants[:8]}")
    print(f"  Words ending in -tor         : {len(suffix_tor)}   e.g. {suffix_tor[:8]}")
    print(f"  Words ending in -val         : {len(suffix_val)}   e.g. {suffix_val[:8]}")
    print(f"  Words ending in -lasi        : {len(suffix_lasi)}  e.g. {suffix_lasi[:8]}")

    # ── 4.2  Train ───────────────────────────────────────────────────────────
    _print_section("STEP 2 — Train BPE Tokenizer")
    tokenizer = GondiTokenizer()
    tokenizer.train(corpus, vocab_size=VOCAB_SIZE, verbose=True)

    # ── 4.3  Inspect top merges ──────────────────────────────────────────────
    _print_section("STEP 3 — Top 30 Merge Rules")
    print(f"  {'#':<5}  {'Pair':<25}  {'New token'}")
    print(f"  {'─'*5}  {'─'*25}  {'─'*20}")
    for i, (a, b) in enumerate(tokenizer.merges[:30], 1):
        print(f"  {i:<5}  '{a}' + '{b}'  →  '{a+b}'")

    # Highlight Gondi-specific merges
    gondi_interest = [
        (a, b) for a, b in tokenizer.merges
        if (a + b) in ("kk", "tt", "gg", "cc", "pp", "nn", "ll", "mm",
                       "tor", "val", "lasi", "ator", "itor",
                       "ator</w>", "val</w>", "lasi</w>", "tor</w>",
                       "al", "an", "or", "ar")
    ]
    print(f"\n  Gondi-specific merges found  : {len(gondi_interest)}")
    for a, b in gondi_interest:
        print(f"    '{a}' + '{b}'  →  '{a+b}'")

    # ── 4.4  Encode & Decode demo ────────────────────────────────────────────
    _print_section("STEP 4 — Encode / Decode Demonstration")

    demo_words = [
        "aggator",    # "man from there" — contains 'gg' and suffix '-ator'
        "vatusval",   # "dry in the sun" — contains suffix '-val'
        # bonus words from the dictionary
        "adenlasi",   # contains suffix '-lasi'
        "aptemayval", # longer compound
        "akkal",      # double-k
        "itteke",     # double-t
    ]

    for word in demo_words:
        tokens = tokenizer.tokenize(word)
        ids    = tokenizer.encode(word)
        back   = tokenizer.decode(ids)

        print(f"\n  Input   : '{word}'")
        print(f"  Tokens  : {tokens}")
        print(f"  IDs     : {ids}")
        print(f"  Decoded : '{back}'")
        print(f"  Round-trip OK: {back == word}")

    # ── 4.5  Save & Reload ───────────────────────────────────────────────────
    _print_section("STEP 5 — Save & Reload")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tokenizer.save(MODEL_PATH)

    reloaded = GondiTokenizer.load(MODEL_PATH)
    print(f"\n  Reloaded tokenizer : {reloaded}")

    # Verify reloaded tokenizer gives identical results
    for word in demo_words:
        orig_ids     = tokenizer.encode(word)
        reloaded_ids = reloaded.encode(word)
        assert orig_ids == reloaded_ids, f"Mismatch on '{word}'"
    print("  All encode outputs match original  ✓")

    # ── 4.6  Focused output for the two required words ───────────────────────
    _print_section("STEP 6 — Required Words: 'aggator' & 'vatusval'")

    for word in ("aggator", "vatusval"):
        tokens = tokenizer.tokenize(word)
        ids    = tokenizer.encode(word)
        back   = tokenizer.decode(ids)

        print(f"\n  ┌─ Word    : '{word}'")
        print(f"  │  Tokens  : {tokens}")
        print(f"  │  IDs     : {ids}")
        print(f"  │  Decoded : '{back}'")
        print(f"  └─ ✓ Round-trip" if back == word else f"  └─ ✗ Round-trip FAILED")

    print("\n" + "═" * 60)
    print("  Done.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
