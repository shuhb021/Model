"""
Inference Engine — Minimum Bayes Risk (MBR) Decoding
──────────────────────────────────────────────────────
Standard beam search produces hallucinations and language-switching
in low-resource regimes. MBR decoding:

    1. Sample K=16 candidates via temperature ancestral sampling (T=0.65)
    2. Score all K×K pairs with chrF utility metric
    3. Select candidate maximising expected utility:
       Ŷ = argmax_{Y ∈ U} (1/K) Σ_j Utility(Y, Y^(j))

Also provides: greedy decoding and beam search (for comparison).
"""

import math
import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─── chrF String Metric ────────────────────────────────────────────────────────

def char_ngrams(text: str, n: int) -> dict:
    """Count character n-grams."""
    ngrams: dict = {}
    for i in range(len(text) - n + 1):
        ng = text[i:i+n]
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


def chrf_score(hypothesis: str, reference: str,
               beta: float = 2.0, max_n: int = 6) -> float:
    """
    chrF score (character n-gram F-score).
    beta=2 weights recall more than precision (standard for MT).
    Returns value in [0, 1].
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0

    precisions, recalls = [], []

    for n in range(1, max_n + 1):
        hyp_ngrams = char_ngrams(hypothesis, n)
        ref_ngrams = char_ngrams(reference,  n)

        if not hyp_ngrams and not ref_ngrams:
            continue

        overlap = sum(min(hyp_ngrams.get(ng, 0), ref_ngrams.get(ng, 0))
                      for ng in set(hyp_ngrams) | set(ref_ngrams))

        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precisions.append(overlap / hyp_total if hyp_total > 0 else 0.0)
        recalls.append(   overlap / ref_total if ref_total > 0 else 0.0)

    if not precisions:
        return 0.0

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls)    / len(recalls)

    if avg_p + avg_r == 0:
        return 0.0

    return (1 + beta**2) * avg_p * avg_r / (beta**2 * avg_p + avg_r)


# ─── Decoding Utilities ────────────────────────────────────────────────────────

def _top_p_sample(logits: torch.Tensor, temperature: float,
                  top_p: float = 0.95) -> int:
    """Nucleus (top-p) sampling with temperature."""
    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Remove tokens once cumulative prob exceeds top_p
    sorted_indices_to_remove = cumulative > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0]  = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    sample_idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[sample_idx].item()


# ─── Inference Engine ─────────────────────────────────────────────────────────

class NMTInference:
    """
    Full inference engine supporting:
    - Greedy decoding
    - Beam search
    - MBR decoding (default, recommended for Gondi)
    """

    def __init__(self, model, tokenizer,
                 translation_memory=None,
                 device: Optional[str] = None,
                 bos_id: int = 2,
                 eos_id: int = 3,
                 pad_id: int = 0):
        self.model  = model
        self.tok    = tokenizer
        self.tm     = translation_memory
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device).eval()

    # ── Encoder ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode(self, src_text: str, src_lang: str):
        ids = self.tok.encode(src_text, lang=src_lang)
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model.encode(src)   # (encoder_output, src_mask)

    # ── Greedy Decode ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(self, encoder_output, src_mask,
                      tgt_lang: str, max_len: int = 256) -> List[int]:
        generated = [self.bos_id]

        for _ in range(max_len):
            tgt = torch.tensor([generated], dtype=torch.long, device=self.device)
            dec_out = self.model.decode(tgt, encoder_output, src_mask)
            logits  = self.model.output_projection(dec_out[:, -1, :])   # [1, V]
            next_id = logits.argmax(dim=-1).item()

            if next_id == self.eos_id:
                break
            generated.append(next_id)

        return generated[1:]   # strip BOS

    # ── Temperature Sampling ──────────────────────────────────────────────────

    @torch.no_grad()
    def sample_decode(self, encoder_output, src_mask,
                      temperature: float = 0.65,
                      max_len: int = 256) -> List[int]:
        generated = [self.bos_id]

        for _ in range(max_len):
            tgt     = torch.tensor([generated], dtype=torch.long, device=self.device)
            dec_out = self.model.decode(tgt, encoder_output, src_mask)
            logits  = self.model.output_projection(dec_out[:, -1, :]).squeeze(0)

            next_id = _top_p_sample(logits, temperature=temperature)
            if next_id == self.eos_id:
                break
            generated.append(next_id)

        return generated[1:]   # strip BOS

    # ── Beam Search ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def beam_search(self, encoder_output, src_mask,
                    beam_size: int = 4,
                    max_len: int = 256,
                    length_penalty: float = 0.6) -> List[int]:
        """
        Standard beam search with length penalty.
        Note: MBR is preferred for Gondi to suppress hallucinations.
        """
        beams = [([self.bos_id], 0.0)]   # (tokens, log_prob)
        completed = []

        for _ in range(max_len):
            candidates = []
            for tokens, score in beams:
                if tokens[-1] == self.eos_id:
                    completed.append((tokens, score))
                    continue

                tgt     = torch.tensor([tokens], dtype=torch.long, device=self.device)
                dec_out = self.model.decode(tgt, encoder_output, src_mask)
                logits  = self.model.output_projection(dec_out[:, -1, :]).squeeze(0)
                log_probs = F.log_softmax(logits, dim=-1)

                top_log_probs, top_ids = log_probs.topk(beam_size)
                for lp, tid in zip(top_log_probs.tolist(), top_ids.tolist()):
                    candidates.append((tokens + [tid], score + lp))

            if not candidates:
                break

            # Apply length penalty and select top beams
            def beam_score(item):
                toks, sc = item
                lp = ((5 + len(toks)) / 6) ** length_penalty
                return sc / lp

            beams = sorted(candidates, key=beam_score, reverse=True)[:beam_size]

        beams = sorted(completed + beams, key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
        best = beams[0][0] if beams else [self.eos_id]

        return [t for t in best if t not in (self.bos_id, self.eos_id)]

    # ── MBR Decoding (recommended) ────────────────────────────────────────────

    @torch.no_grad()
    def mbr_decode(self, encoder_output, src_mask,
                   k: int = 16,
                   temperature: float = 0.65,
                   max_len: int = 256) -> List[int]:
        """
        Minimum Bayes Risk decoding:
        Ŷ = argmax_{Y ∈ U} (1/K) Σ_j chrF(Y, Y^(j))
        
        Generates K=16 candidates, scores all pairs, returns
        the candidate with highest consensus utility.
        """
        # Step 1: Generate K candidate translations
        candidates_ids = []
        candidates_str = []

        for _ in range(k):
            ids = self.sample_decode(encoder_output, src_mask,
                                     temperature=temperature,
                                     max_len=max_len)
            text = self.tok.decode(ids)
            candidates_ids.append(ids)
            candidates_str.append(text)

        if not candidates_str:
            return []

        # Step 2: Pairwise chrF utility scoring
        utilities = []
        for i, hyp in enumerate(candidates_str):
            util = sum(
                chrf_score(hyp, candidates_str[j])
                for j in range(k) if j != i
            ) / max(k - 1, 1)
            utilities.append(util)

        # Step 3: Return highest-utility candidate
        best_idx = max(range(len(utilities)), key=lambda i: utilities[i])
        return candidates_ids[best_idx]

    # ── Main Translate API ────────────────────────────────────────────────────

    def translate(self, text: str,
                  src_lang: str,
                  tgt_lang: str,
                  method: str = "mbr",
                  use_tm: bool = True,
                  **kwargs) -> str:
        """
        Main translation entry point.

        Args:
            text     : Source sentence
            src_lang : 'gondi', 'hindi', or 'english'
            tgt_lang : 'gondi', 'hindi', or 'english'
            method   : 'mbr' (default) | 'beam' | 'greedy'
            use_tm   : Whether to use Translation Memory prefix fusion
        """
        # TM fusion prefix
        input_text = text
        if use_tm and self.tm is not None:
            lang_tag = f"__{tgt_lang[:3].lower()}_{'Deva' if tgt_lang == 'hindi' else 'Latn'}__"
            prefix = self.tm.get_fusion_prefix(text, lang_tag)
            if prefix:
                input_text = prefix

        # Encode
        encoder_output, src_mask = self._encode(input_text, src_lang)

        # Decode
        if method == "mbr":
            ids = self.mbr_decode(encoder_output, src_mask, **kwargs)
        elif method == "beam":
            ids = self.beam_search(encoder_output, src_mask, **kwargs)
        else:
            ids = self.greedy_decode(encoder_output, src_mask, tgt_lang=tgt_lang)

        return self.tok.decode(ids)

    def translate_batch(self, texts: List[str],
                        src_lang: str, tgt_lang: str,
                        method: str = "mbr") -> List[str]:
        return [self.translate(t, src_lang, tgt_lang, method=method)
                for t in texts]


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test chrF metric
    hyp = "The cat sat on the mat"
    ref = "The cat sat on a mat"
    score = chrf_score(hyp, ref)
    print(f"chrF score: {score:.4f}  (expected ~0.85+)")

    hyp2 = "Random totally different text"
    score2 = chrf_score(hyp2, ref)
    print(f"chrF score: {score2:.4f}  (expected ~0.1-)")

    # Simulate MBR selection
    candidates = [
        "main ghar jata hoon",
        "main ghar jata hun",
        "main ghar ko jata hoon",
        "random unrelated output",
    ]
    utilities = []
    k = len(candidates)
    for i, hyp in enumerate(candidates):
        u = sum(chrf_score(hyp, candidates[j]) for j in range(k) if j != i) / (k-1)
        utilities.append(u)
        print(f"  [{i}] {hyp:<35} | utility={u:.4f}")

    best = max(range(k), key=lambda i: utilities[i])
    print(f"\n[OK] MBR selected: [{best}] '{candidates[best]}'")
