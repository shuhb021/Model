"""
Custom Asymmetric Multilingual NMT Architecture
For Low-Resource Tribal Indic Languages (Gondi ↔ Hindi ↔ English)
~137M Parameters | Asymmetric 8-Enc / 4-Dec | T5 Relative Bias | Gated-GeLU FFN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─── Relative Position Bias (T5-style) ────────────────────────────────────────

class RelativePositionBias(nn.Module):
    """
    T5-style learned relative position bias.
    Biases are shared across all layers but learned per attention head.
    Uses logarithmic bucketing to handle long sequences efficiently.
    """

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor,
                                   bidirectional: bool = True,
                                   num_buckets: int = 32,
                                   max_distance: int = 128) -> torch.Tensor:
        """Map relative positions to bucket indices using logarithmic scale."""
        ret = 0
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large,
                                  torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_length: int, key_length: int,
                device: torch.device) -> torch.Tensor:
        """
        Returns: [1, num_heads, query_length, key_length]
        """
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position  = torch.arange(key_length,   dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # [q, k]

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)       # [q, k, heads]
        values = values.permute(2, 0, 1).unsqueeze(0)          # [1, heads, q, k]
        return values


# ─── Gated-GeLU Feed-Forward Network ──────────────────────────────────────────

class GatedGeLUFFN(nn.Module):
    """
    Gated Linear Unit with GeLU activation for morphology-aware routing.
    FFN_gated-gelu(x) = (GeLU(x × W_g) ⊗ (x × W_1)) × W_2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # primary path
        self.Wg = nn.Linear(d_model, d_ff, bias=False)   # gate path
        self.W2 = nn.Linear(d_ff, d_model, bias=False)   # output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate   = F.gelu(self.Wg(x))     # GeLU gate
        linear = self.W1(x)              # linear branch
        gated  = gate * linear           # Hadamard product
        return self.W2(self.dropout(gated))


# ─── Multi-Head Attention with Relative Position Bias ─────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional T5 relative position bias injection.
    A_ij = (Q_i × K_j^T) / sqrt(d_k) + β_ij
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * D)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        Q = self._split_heads(self.q_proj(query))   # [B, H, S, d_k]
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, S_q, S_k]

        # Inject T5 relative position bias: β_ij
        if position_bias is not None:
            attn_scores = attn_scores + position_bias

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)     # [B, H, S_q, d_k]
        return self.out_proj(self._merge_heads(context))


# ─── Encoder Layer ─────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """Single transformer encoder layer with Pre-LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.15):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = GatedGeLUFFN(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention + residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask=src_mask, position_bias=position_bias)
        x = residual + self.dropout(x)

        # Pre-norm FFN + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


# ─── Decoder Layer ─────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """Single transformer decoder layer: masked self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.15):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = GatedGeLUFFN(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                self_position_bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Masked self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask=tgt_mask, position_bias=self_position_bias)
        x = residual + self.dropout(x)

        # Cross-attention (encoder-decoder)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = residual + self.dropout(x)

        # Gated-GeLU FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


# ─── Encoder Stack ─────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Deeper encoder (L_enc=8) for robust cross-lingual feature extraction.
    Shared relative position bias across all layers.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, dropout: float = 0.15,
                 num_buckets: int = 32, max_distance: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pos_bias = RelativePositionBias(num_heads, num_buckets, max_distance)

    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[1]
        position_bias = self.pos_bias(seq_len, seq_len, x.device)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask, position_bias=position_bias)
        return self.norm(x)


# ─── Decoder Stack ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Shallow decoder (L_dec=4) to minimize autoregressive compounding errors.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, dropout: float = 0.15,
                 num_buckets: int = 32, max_distance: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pos_bias = RelativePositionBias(num_heads, num_buckets, max_distance)

    def forward(self, x: torch.Tensor,
                encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt_len = x.shape[1]
        self_position_bias = self.pos_bias(tgt_len, tgt_len, x.device)

        for layer in self.layers:
            x = layer(x, encoder_output,
                      tgt_mask=tgt_mask,
                      src_mask=src_mask,
                      self_position_bias=self_position_bias)
        return self.norm(x)


# ─── Full NMT Model ────────────────────────────────────────────────────────────

class AsymmetricNMT(nn.Module):
    """
    Custom Asymmetric Multilingual NMT Model
    ─────────────────────────────────────────
    Architecture:
        • Vocabulary  : 32,000 (shared multilingual BPE)
        • d_model     : 512
        • d_ff        : 2,048 (Gated-GeLU)
        • Encoder     : 8 layers  (deeper for representation)
        • Decoder     : 4 layers  (shallower for fast inference)
        • Heads       : 8  (d_k = 64)
        • Pos. Enc.   : T5 Relative Bias (shared, per-head learned)
        • Dropout     : 0.15
        • Parameters  : ~137M total
    """

    def __init__(self,
                 vocab_size: int    = 32_000,
                 d_model: int       = 512,
                 d_ff: int          = 2_048,
                 num_heads: int     = 8,
                 enc_layers: int    = 8,
                 dec_layers: int    = 4,
                 dropout: float     = 0.15,
                 max_seq_len: int   = 256,
                 pad_id: int        = 0):
        super().__init__()
        self.d_model    = d_model
        self.pad_id     = pad_id
        self.vocab_size = vocab_size

        # Shared embedding (encoder + decoder input)
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_scale  = math.sqrt(d_model)
        self.dropout    = nn.Dropout(dropout)

        self.encoder = Encoder(d_model, num_heads, d_ff, enc_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, dec_layers, dropout)

        # Output projection — tied to embedding weights
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask: [B, 1, 1, S]"""
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Causal + padding mask: [B, 1, T, T]"""
        T = tgt.shape[1]
        pad_mask   = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)   # [B,1,1,T]
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # [T,T]
        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = self._make_src_mask(src)
        x = self.dropout(self.embedding(src) * self.emb_scale)
        encoder_output = self.encoder(x, src_mask)
        return encoder_output, src_mask

    def decode(self, tgt: torch.Tensor,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = self._make_tgt_mask(tgt)
        x = self.dropout(self.embedding(tgt) * self.emb_scale)
        return self.decoder(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src : [B, S]  — source token ids
            tgt : [B, T]  — target token ids (teacher-forced, shifted right)
        Returns:
            logits : [B, T, V]
        """
        encoder_output, src_mask = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output, src_mask)
        return self.output_projection(decoder_output)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = AsymmetricNMT()
    total = model.count_parameters()
    print(f"[OK] Model initialised | Parameters: {total:,} (~{total/1e6:.1f}M)")

    B, S, T = 2, 32, 28
    src = torch.randint(1, 32000, (B, S))
    tgt = torch.randint(1, 32000, (B, T))

    with torch.no_grad():
        logits = model(src, tgt)
    print(f"[OK] Forward pass OK   | logits shape: {tuple(logits.shape)}")
