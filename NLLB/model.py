import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ModelConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        d_ff: int = 2048,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.15,
        max_seq_len: int = 256,
        r: int = 16,
        alpha: int = 32,
        use_lora: bool = False
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.r = r
        self.alpha = alpha
        self.use_lora = use_lora

# ─────────────────────────────────────────────────────────────────────────────
# 1. T5 RELATIVE POSITION BIAS
# ─────────────────────────────────────────────────────────────────────────────

class T5RelativePositionBias(nn.Module):
    """
    T5 relative position bias module.
    Maps relative positions to buckets log-spaced for long distances.
    """
    def __init__(self, num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True, num_heads: int = 8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Maps relative position offsets to bucket indices.
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # Now n >= 0
        # Half of the buckets are for exact small distances
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Logarithmic spacing for larger distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> torch.Tensor:
        """
        Computes the bias matrix.
        Returns: Tensor of shape (1, num_heads, seq_len_q, seq_len_k)
        """
        context_position = torch.arange(seq_len_q, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len_k, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        
        buckets = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        # Shape: (seq_len_q, seq_len_k, num_heads)
        bias = self.relative_attention_bias(buckets)
        # Permute to: (1, num_heads, seq_len_q, seq_len_k)
        bias = bias.permute([2, 0, 1]).unsqueeze(0)
        return bias

# ─────────────────────────────────────────────────────────────────────────────
# 2. GATED-GELU FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class GatedGeLUFeedForward(nn.Module):
    """
    Gated-GeLU FFN as defined in Shazeer (2020) "GLU Variants Improve Transformer".
    FFN_{Gated-GeLU}(x) = (GeLU(x W_g) * (x W_1)) W_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)  # W_g (gated branch)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)  # W_1 (linear branch)
        self.wo = nn.Linear(d_ff, d_model, bias=False)    # W_2 (output projection)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute branches
        gate = F.gelu(self.wi_0(x))
        linear_branch = self.wi_1(x)
        # Element-wise product
        gated_out = gate * linear_branch
        gated_out = self.dropout(gated_out)
        # Final projection
        return self.wo(gated_out)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LORA ADAPTER LAYER
# ─────────────────────────────────────────────────────────────────────────────

class LoraLinear(nn.Module):
    """
    LoRA parameter-efficient adapter wrapper around a standard nn.Linear.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros((r, base_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r)))
        
        # Initialize LoRA weights (lora_A: Kaiming uniform, lora_B: zeros)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze the base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard projection
        base_out = self.base_layer(x)
        # LoRA pathway
        lora_out = F.linear(x, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out

def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32):
    """
    Recursively replaces target query, key, value, and output linear projections
    inside self-attention blocks with LoraLinear wrappers.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and any(tgt in name for tgt in ["q_proj", "k_proj", "v_proj", "out_proj"]):
            setattr(model, name, LoraLinear(child, r=r, alpha=alpha))
        else:
            inject_lora(child, r=r, alpha=alpha)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-HEAD SELF & CROSS-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self/Cross Attention module supporting relative position bias.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Linear projections & split into heads
        # Shape: (B, H, S, d_k)
        queries = self.q_proj(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        # Shape: (B, H, S_q, S_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Inject relative position bias if provided
        if position_bias is not None:
            scores = scores + position_bias

        # Apply padding / causal masking
        if mask is not None:
            # mask shape: (B, 1, 1, S_k) or (B, 1, S_q, S_k)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context output
        context = torch.matmul(attn_weights, values)
        # Reshape context back to (B, S_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        output = self.out_proj(context)
        return self.proj_dropout(output)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ASYMMETRIC ENCODER & DECODER LAYERS
# ─────────────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Transformer Encoder layer.
    Pre-LayerNorm layout.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GatedGeLUFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=mask, position_bias=position_bias)
        x = x + attn_out

        # FFN with pre-norm
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out
        return x

class DecoderLayer(nn.Module):
    """
    Transformer Decoder layer.
    Pre-LayerNorm layout with Cross-Attention.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = GatedGeLUFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_outputs: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        self_position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Masked self-attention
        norm_x = self.norm1(x)
        self_attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=self_mask, position_bias=self_position_bias)
        x = x + self_attn_out

        # Cross-attention (no position bias)
        norm_x2 = self.norm2(x)
        cross_attn_out = self.cross_attn(norm_x2, enc_outputs, enc_outputs, mask=cross_mask)
        x = x + cross_attn_out

        # FFN
        norm_x3 = self.norm3(x)
        ffn_out = self.ffn(norm_x3)
        x = x + ffn_out
        return x

# ─────────────────────────────────────────────────────────────────────────────
# 6. ASYMMETRIC TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CustomNllbModel(nn.Module):
    """
    The full asymmetric transformer architecture containing:
    - Shared embeddings.
    - Deep Encoder (L_enc layers).
    - Shallow Decoder (L_dec layers).
    - Relative position bias layers.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Shared token embedding
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)

        # Position Bias Generators (one bidirectional for encoder, one causal for decoder)
        self.enc_pos_bias = T5RelativePositionBias(bidirectional=True, num_heads=config.num_heads)
        self.dec_pos_bias = T5RelativePositionBias(bidirectional=False, num_heads=config.num_heads)

        # Encoder stack (e.g. 8 layers)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.enc_norm = nn.LayerNorm(config.d_model)

        # Decoder stack (e.g. 4 layers)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.dec_norm = nn.LayerNorm(config.d_model)

        # Final projection head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights between embedding and final lm_head as in standard NLLB/T5
        self.lm_head.weight = self.embeddings.weight

        # Apply LoRA if configured
        if config.use_lora:
            inject_lora(self, r=config.r, alpha=config.alpha)

    def encode(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = src_ids.device
        seq_len = src_ids.shape[1]

        # Convert token IDs to embeddings
        x = self.dropout(self.embeddings(src_ids) * math.sqrt(self.config.d_model))

        # Compute relative position bias
        pos_bias = self.enc_pos_bias(seq_len, seq_len, device)

        # Forward encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask, position_bias=pos_bias)
        
        return self.enc_norm(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        enc_outputs: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = tgt_ids.device
        seq_len = tgt_ids.shape[1]

        # Convert token IDs to embeddings
        x = self.dropout(self.embeddings(tgt_ids) * math.sqrt(self.config.d_model))

        # Compute relative position bias
        pos_bias = self.dec_pos_bias(seq_len, seq_len, device)

        # Forward decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                enc_outputs,
                self_mask=self_mask,
                cross_mask=cross_mask,
                self_position_bias=pos_bias
            )
        
        x = self.dec_norm(x)
        return self.lm_head(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full sequence forward pass.
        src_mask: shape (Batch, 1, 1, SeqLen_src)
        tgt_mask: shape (Batch, 1, SeqLen_tgt, SeqLen_tgt) indicating causal mask + padding
        """
        # 1. Encode source
        enc_outputs = self.encode(src_ids, src_mask)

        # 2. Decode to target vocab probabilities
        # Setup cross attention mask: shape should align to (Batch, 1, 1, SeqLen_src) to mask source padding
        logits = self.decode(
            tgt_ids,
            enc_outputs,
            self_mask=tgt_mask,
            cross_mask=src_mask
        )
        return logits

# ─────────────────────────────────────────────────────────────────────────────
# 7. UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (Total parameters, Trainable parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    # Test compilation and forward pass
    cfg = ModelConfig(vocab_size=32000, use_lora=True)
    model = CustomNllbModel(cfg)
    
    total, trainable = count_parameters(model)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters (with LoRA): {trainable:,}")
    
    # Dummy tensors
    # Batch size = 2, Sequence length = 10
    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 8))
    
    # Masks (1 for valid token, 0 for pad)
    s_mask = torch.ones((2, 1, 1, 10))
    # Causal lower triangular mask combined with padding mask
    t_mask = torch.tril(torch.ones((8, 8))).view(1, 1, 8, 8)
    
    out = model(src, tgt, src_mask=s_mask, tgt_mask=t_mask)
    print("Forward pass successful. Output shape:", out.shape)
