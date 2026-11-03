"""
dataset.py — Unified Parallel Data Loading for Gondi Translation Baselines
===========================================================================
Supports both:
  1. HuggingFace BPE tokenizers (tokenizer.json, e.g. 1157 BPE tokens used by ByT5)
  2. Native Gondi BPE tokenizers (gondi_bpe_model.json, e.g. 5000 subwords)
"""

import os
import json
from typing import Tuple, List, Optional, Union
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class UnifiedBPETokenizer:
    """
    Unified wrapper for both HuggingFace Fast Tokenizer and native GondiTokenizer.
    Guarantees consistent API across all baseline experiments.
    """
    def __init__(self, path: str):
        self.path = path
        is_hf = False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "model" in data and ("type" in data["model"] or "version" in data):
                    is_hf = True
        except Exception:
            pass

        if is_hf:
            from transformers import PreTrainedTokenizerFast
            self.hf_tok = PreTrainedTokenizerFast(
                tokenizer_file=path,
                unk_token="[UNK]",
                pad_token="[PAD]",
                eos_token="</s>",
                bos_token="[START]"
            )
            self.is_hf = True
            self.pad_idx = self.hf_tok.pad_token_id if self.hf_tok.pad_token_id is not None else 1
            self.unk_idx = self.hf_tok.unk_token_id if self.hf_tok.unk_token_id is not None else 0
            self.eos_idx = self.hf_tok.eos_token_id if self.hf_tok.eos_token_id is not None else 2
            self.sos_idx = self.hf_tok.bos_token_id if self.hf_tok.bos_token_id is not None else 3
            self._vocab_size = len(self.hf_tok)
            self.id_to_token = {v: k for k, v in self.hf_tok.get_vocab().items()}
        else:
            from gondi_bpe_tokenizer import GondiTokenizer
            self.custom_tok = GondiTokenizer.load(path)
            self.is_hf = False
            self.pad_idx = self.custom_tok.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
            self.unk_idx = self.custom_tok.token_to_id.get(GondiTokenizer.UNK_TOKEN, 1)
            self.sos_idx = self.custom_tok.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
            self.eos_idx = self.custom_tok.token_to_id.get(GondiTokenizer.END_TOKEN, 3)
            self._vocab_size = self.custom_tok.vocab_size()
            self.id_to_token = self.custom_tok.id_to_token

    def encode(self, text: str) -> List[int]:
        if self.is_hf:
            return self.hf_tok.encode(text, add_special_tokens=False)
        else:
            return self.custom_tok.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        if self.is_hf:
            return self.hf_tok.decode(token_ids, skip_special_tokens=True)
        else:
            tokens = []
            for tid in token_ids:
                if tid == self.eos_idx:
                    break
                if tid not in (self.pad_idx, self.sos_idx):
                    tok = self.custom_tok.id_to_token.get(tid, "")
                    if tok:
                        tokens.append(tok)
            return "".join(tokens).replace(" ", " ").strip()

    def vocab_size(self) -> int:
        return self._vocab_size


def load_unified_tokenizer(tokenizer_path: str) -> UnifiedBPETokenizer:
    return UnifiedBPETokenizer(tokenizer_path)


class GondiTranslationDataset(Dataset):
    """
    Parallel Translation Dataset for Gondi -> Target language (Hindi / English).
    Uses the unified BPE tokenizer for both source and target.
    """
    def __init__(self, csv_path: str, tokenizer: UnifiedBPETokenizer,
                 src_col: str = "Gondi", tgt_col: str = "Hindi",
                 max_samples: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_idx
        self.sos_idx = tokenizer.sos_idx
        self.eos_idx = tokenizer.eos_idx

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.dropna(subset=[src_col, tgt_col]).reset_index(drop=True)

        df[src_col] = df[src_col].astype(str).str.strip()
        df[tgt_col] = df[tgt_col].astype(str).str.strip()
        df = df[(df[src_col] != "") & (df[tgt_col] != "")].reset_index(drop=True)

        if max_samples and max_samples < len(df):
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.src_texts = df[src_col].tolist()
        self.tgt_texts = df[tgt_col].tolist()

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)

        src_tensor = torch.tensor([self.sos_idx] + src_ids + [self.eos_idx], dtype=torch.long)
        tgt_tensor = torch.tensor([self.sos_idx] + tgt_ids + [self.eos_idx], dtype=torch.long)

        return src_tensor, tgt_tensor


def pad_collate_fn(batch, pad_idx: int = 0):
    src_seqs, tgt_seqs = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded, src_lengths
