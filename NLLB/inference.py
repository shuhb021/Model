import torch
import torch.nn.functional as F
import collections
import re
import sys
from typing import List, Tuple, Optional
from model import CustomNllbModel, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. PURE PYTHON CHRF SCORER
# ─────────────────────────────────────────────────────────────────────────────

def get_char_ngrams(s: str, n: int) -> List[str]:
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def calculate_chrf(candidate: str, reference: str, n_max: int = 6, beta: float = 2.0) -> float:
    """
    Computes chrF score (Character n-gram F-score) between a candidate and a reference.
    Standard beta is 2.0 (recall is weighted twice as much as precision).
    """
    c_clean = re.sub(r"\s+", "", candidate)  # chrF typically counts without whitespace or handles it
    r_clean = re.sub(r"\s+", "", reference)
    
    if not c_clean or not r_clean:
        return 0.0
        
    precisions = []
    recalls = []
    
    for n in range(1, n_max + 1):
        c_ngrams = get_char_ngrams(c_clean, n)
        r_ngrams = get_char_ngrams(r_clean, n)
        
        if not c_ngrams or not r_ngrams:
            continue
            
        c_counts = collections.Counter(c_ngrams)
        r_counts = collections.Counter(r_ngrams)
        
        # Intersection of counts
        matches = sum((c_counts & r_counts).values())
        
        p = matches / len(c_ngrams) if len(c_ngrams) > 0 else 0.0
        r = matches / len(r_ngrams) if len(r_ngrams) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        
    if not precisions:
        return 0.0
        
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    
    if (beta**2 * avg_p) + avg_r == 0:
        return 0.0
        
    f_score = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
    return f_score

# ─────────────────────────────────────────────────────────────────────────────
# 2. AUTOREGRESSIVE DECODING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sequence(
    model: CustomNllbModel,
    src_ids: torch.Tensor,
    max_len: int = 64,
    temperature: float = 1.0,
    do_sample: bool = False,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Performs autoregressive token generation.
    Supports greedy search and temperature-scaled sampling.
    """
    model.eval()
    device = src_ids.device
    batch_size = src_ids.shape[0]

    # 1. Encode source
    # Create attention mask for source padding (if any)
    src_mask = (src_ids != pad_token_id).unsqueeze(1).unsqueeze(2).to(torch.float32)
    enc_outputs = model.encode(src_ids, src_mask)

    # 2. Initialize target sequence with BOS / target language tag
    # shape: (Batch, CurrentLen)
    tgt_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

    for step in range(max_len):
        # Create causal mask for target self-attention
        curr_len = tgt_ids.shape[1]
        tgt_mask = torch.tril(torch.ones((curr_len, curr_len), device=device)).view(1, 1, curr_len, curr_len)
        
        # Decode next token distribution
        logits = model.decode(
            tgt_ids,
            enc_outputs,
            self_mask=tgt_mask,
            cross_mask=src_mask
        )
        
        # Get logits of the last token in sequence
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature and sampling/greedy
        if do_sample:
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        # Append next token to target sequence
        tgt_ids = torch.cat([tgt_ids, next_token], dim=-1)
        
        # If all sequences generated EOS, stop early
        if (tgt_ids == eos_token_id).any(dim=-1).all():
            break
            
    return tgt_ids

# ─────────────────────────────────────────────────────────────────────────────
# 3. MINIMUM BAYES RISK (MBR) DECODING
# ─────────────────────────────────────────────────────────────────────────────

class MbrDecoder:
    """
    Implements Minimum Bayes Risk (MBR) decoding to reduce translations hallucinations
    by selecting the consensus translation from a candidate sample pool.
    """
    def __init__(self, model: CustomNllbModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def decode_mbr(
        self,
        src_text: str,
        pool_size: int = 16,
        temperature: float = 0.7,
        max_len: int = 64,
        device: torch.device = torch.device("cpu")
    ) -> str:
        # Tokenize source
        src_tokens = self.tokenizer.encode(src_text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        
        # Generate target language tags or starting tokens (BOS)
        bos_id = self.tokenizer.token_to_id.get("[START]", 2)
        eos_id = self.tokenizer.token_to_id.get("[END]", 3)
        pad_id = self.tokenizer.token_to_id.get("[PAD]", 0)

        # 1. Generate candidate pool (using sampling)
        candidates = []
        for _ in range(pool_size):
            gen_ids = generate_sequence(
                self.model,
                src_tensor,
                max_len=max_len,
                temperature=temperature,
                do_sample=True,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id
            )
            # Convert token IDs to string
            candidate_text = self.tokenizer.decode(gen_ids[0].tolist())
            candidates.append(candidate_text)
            
        # Remove duplicates from pool to save comparisons
        unique_candidates = list(set(candidates))
        if len(unique_candidates) == 1:
            return unique_candidates[0]
            
        # 2. Compute pairwise similarity scores (expected utility)
        best_candidate = unique_candidates[0]
        best_score = -1.0
        
        for cand in unique_candidates:
            total_score = 0.0
            for ref in candidates:  # Reference pool preserves the frequency weight of generation
                total_score += calculate_chrf(cand, ref)
            
            avg_score = total_score / len(candidates)
            if avg_score > best_score:
                best_score = avg_score
                best_candidate = cand
                
        return best_candidate

# ─────────────────────────────────────────────────────────────────────────────
# 4. POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class PostProcessor:
    """
    Handles script normalization, Named Entity Protection, and trailing formatting.
    """
    def __init__(self):
        pass

    def process(self, text: str) -> str:
        # 1. Clean extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        # 2. Punctuation normalization
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
        # 3. Capitalization for English translation direction
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        return text

if __name__ == "__main__":
    # Avoid print encoding crashes on Windows PowerShell
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    # Quick verification tests
    c1 = "गोंडी भाषा बहुत पुरानी है"
    c2 = "गोंडी भाषा बहुत पुरानी है"
    c3 = "यह गोंडी बोली प्राचीन है"
    
    print("chrF Match score c1-c2:", calculate_chrf(c1, c2))
    print("chrF Match score c1-c3:", calculate_chrf(c1, c3))
    
    # Test post processor
    pp = PostProcessor()
    print("Post-processed:", pp.process("  this is a test .   "))
