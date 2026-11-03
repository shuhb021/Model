"""
Multilingual BPE Tokenizer Wrapper
────────────────────────────────────
Joint 32K SentencePiece/BPE vocabulary with NLLB-style language tags:
    __gon_Latn__  → Gondi (Latin script, transliterated)
    __hin_Deva__  → Hindi (Devanagari)
    __eng_Latn__  → English (Latin)

Handles:
    • Script-level normalization (Devanagari Gondi → Latin)
    • Language tag prepending
    • BPE-Dropout (p=0.1) for subword regularization during training
"""

import unicodedata
from pathlib import Path
from typing import List, Optional, Union

try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False
    print("⚠️  sentencepiece not installed. Install with: pip install sentencepiece")

# ─── Language Tags ─────────────────────────────────────────────────────────────

LANG_TAGS = {
    "gondi":   "__gon_Latn__",
    "hindi":   "__hin_Deva__",
    "english": "__eng_Latn__",
    "gon":     "__gon_Latn__",
    "hin":     "__hin_Deva__",
    "eng":     "__eng_Latn__",
}

SPECIAL_TOKENS = {
    "pad":    "<pad>",
    "unk":    "<unk>",
    "bos":    "<s>",
    "eos":    "</s>",
    "sep":    "<sep>",
    "tm_start": "[TM_START]",
    "tm_sep":   "[TM_SEP]",
    "tm_end":   "[TM_END]",
}

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ─── Script Normalisation ──────────────────────────────────────────────────────

# Simple Devanagari → Latin transliteration table for Gondi
# (Full ITRANS/ISO 15919 mapping should be used in production)
DEVA_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    '।': '.', 'ं': 'm', 'ः': 'h', '्': '',
}


def normalize_text(text: str, lang: str = "hindi") -> str:
    """
    Apply script-level normalization:
    - NFC unicode normalization
    - Devanagari → Latin for Gondi (if script is Devanagari-written Gondi)
    """
    text = unicodedata.normalize("NFC", text.strip())

    if lang in ("gondi", "gon"):
        # Only transliterate if text contains Devanagari characters
        if any('\u0900' <= c <= '\u097F' for c in text):
            result = []
            for ch in text:
                result.append(DEVA_TO_LATIN.get(ch, ch))
            text = "".join(result)

    return text


# ─── Tokenizer Class ───────────────────────────────────────────────────────────

class MultilingualTokenizer:
    """
    Wrapper around SentencePiece model providing:
    - Language-tag prepending
    - BPE-Dropout for training-time regularization
    - TM fusion prefix formatting
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.sp_model = None
        self.vocab_size = 32_000

        if model_path and SPM_AVAILABLE:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(str(model_path))
            self.vocab_size = self.sp_model.GetPieceSize()
            print(f"✅ Loaded SPM model: {model_path} | vocab={self.vocab_size:,}")
        else:
            print("ℹ️  Running in mock mode (no SPM model path given). "
                  "Provide model_path= to a trained .model file for real tokenization.")

    def _lang_tag(self, lang: str) -> str:
        key = lang.lower()
        if key not in LANG_TAGS:
            raise ValueError(f"Unknown language '{lang}'. Valid: {list(LANG_TAGS.keys())}")
        return LANG_TAGS[key]

    def encode(self, text: str, lang: str,
               add_bos: bool = True, add_eos: bool = True,
               bpe_dropout: float = 0.0) -> List[int]:
        """
        Tokenize a sentence with language tag.

        Args:
            text        : Raw input string
            lang        : Language code ('gondi', 'hindi', 'english', etc.)
            add_bos     : Prepend BOS token
            add_eos     : Append EOS token
            bpe_dropout : BPE-Dropout probability (use 0.1 during training)

        Returns:
            List of token ids
        """
        text = normalize_text(text, lang)
        lang_tag = self._lang_tag(lang)
        tagged_text = f"{lang_tag} {text}"

        if self.sp_model is None:
            # Mock tokenization for testing
            ids = [hash(t) % (self.vocab_size - 10) + 4
                   for t in tagged_text.split()]
        elif bpe_dropout > 0.0 and SPM_AVAILABLE:
            ids = self.sp_model.SampleEncode(tagged_text, -1, bpe_dropout)
        else:
            ids = self.sp_model.Encode(tagged_text)

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]

        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to string, stripping special tokens."""
        # Remove special token ids
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]

        if self.sp_model is None:
            return f"<mock_decoded:{len(ids)}_tokens>"
        return self.sp_model.Decode(ids)

    def batch_encode(self, texts: List[str], lang: str,
                     max_length: int = 256,
                     bpe_dropout: float = 0.0,
                     pad: bool = True):
        """
        Encode a batch of sentences, with optional padding and truncation.

        Returns:
            input_ids  : List[List[int]]
            attention_mask : List[List[int]]
        """
        encoded = [
            self.encode(t, lang, bpe_dropout=bpe_dropout)[:max_length]
            for t in texts
        ]

        if not pad:
            return {"input_ids": encoded,
                    "attention_mask": [[1]*len(e) for e in encoded]}

        max_len = max(len(e) for e in encoded)
        input_ids      = [e + [PAD_ID] * (max_len - len(e)) for e in encoded]
        attention_mask = [[1]*len(e) + [0]*(max_len - len(e)) for e in encoded]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def format_tm_prefix(self, tm_src: str, tm_tgt: str,
                         query_src: str, tgt_lang: str) -> str:
        """
        Construct TM-augmented prefix for retrieval-augmented generation:
        [TM_START] gondi_src [TM_SEP] hindi_tgt [TM_END] [LANG_TAG] query_source
        """
        lang_tag = self._lang_tag(tgt_lang)
        return (f"{SPECIAL_TOKENS['tm_start']} {tm_src} "
                f"{SPECIAL_TOKENS['tm_sep']} {tm_tgt} "
                f"{SPECIAL_TOKENS['tm_end']} {lang_tag} {query_src}")


# ─── Training Script Generator ─────────────────────────────────────────────────

SPMTRAIN_CMD = """
# Train a joint multilingual BPE model with SentencePiece
# Run this command after collecting your combined corpus:

python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    input='data/combined_corpus.txt',      # one sentence per line, all languages
    model_prefix='configs/multilingual_bpe',
    vocab_size=32000,
    character_coverage=0.9995,             # high coverage for Devanagari
    model_type='bpe',
    byte_fallback=True,                    # handles OOV bytes gracefully
    user_defined_symbols=[
        '__gon_Latn__', '__hin_Deva__', '__eng_Latn__',
        '[TM_START]', '[TM_SEP]', '[TM_END]',
        '<pad>', '<sep>',
    ],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    shuffle_input_sentence=True,
    input_sentence_size=5000000,
)
"
echo "SPM model saved to configs/multilingual_bpe.model"
"""

if __name__ == "__main__":
    print("─── Tokenizer Smoke Test ───")
    tok = MultilingualTokenizer()  # mock mode

    hi_ids = tok.encode("मैं घर जाता हूँ", lang="hindi")
    en_ids = tok.encode("The quick brown fox", lang="english")
    gon_ids = tok.encode("tor lasi gondi text", lang="gondi")

    print(f"Hindi   ids ({len(hi_ids)}): {hi_ids[:8]}...")
    print(f"English ids ({len(en_ids)}): {en_ids[:8]}...")
    print(f"Gondi   ids ({len(gon_ids)}): {gon_ids[:8]}...")

    prefix = tok.format_tm_prefix(
        tm_src="maa dhanisang", tm_tgt="मेरा घर",
        query_src="anna dhanisang", tgt_lang="hindi"
    )
    print(f"\nTM Prefix: {prefix}")
    print(f"\nSPM Training Command:\n{SPMTRAIN_CMD}")
