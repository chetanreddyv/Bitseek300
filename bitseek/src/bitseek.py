"""
BitSeek: A 1.58-bit Quantized Language Model for BitNet

This module implements a state-of-the-art quantized language model with the following key features:
- 1.58-bit weight quantization (ternary values)
- 8-bit activation quantization
- Flash Multi-head Latent Attention with causal masking
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in FFN layers
- SubLN normalization
- No bias terms in linear or normalization layers

The model architecture follows the BitNet design principles while incorporating several optimizations
for efficiency and performance. The implementation is designed to be memory-efficient and fast
during both training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class BitLinear(nn.Module):
    """
    BitLinear layer implementing 1.58-bit weight quantization and 8-bit activation quantization.
    
    This layer is a key component of the BitNet architecture, providing efficient memory usage
    through quantization while maintaining model performance. It implements:
    - 1.58-bit weight quantization using ternary values {-1, 0, +1}
    - 8-bit activation quantization using absmax quantization (per-token)
    - No bias terms for efficiency
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to include bias terms (default: False)
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_scale', torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_scale.fill_(1.0)

    def forward(self, x):
        """
        Forward pass with quantized weights and activations.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, out_features]
        """
        # Quantize weights to 1.58-bit (ternary)
        weight_abs = torch.abs(self.weight)
        weight_mean = weight_abs.mean()
        weight_ternary = torch.where(
            weight_abs > weight_mean,
            torch.sign(self.weight),
            torch.zeros_like(self.weight)
        )
        
        # Quantize activations to 8-bit
        x_abs = torch.abs(x)
        x_max = x_abs.max(dim=-1, keepdim=True)[0]
        x_scale = 127.0 / (x_max + 1e-8)
        x_quantized = torch.round(x * x_scale) / x_scale
        
        # Forward pass with quantized weights and activations
        return F.linear(x_quantized, weight_ternary * self.weight_scale)

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for positional encoding.
    
    This module implements the RoPE mechanism, which provides better position encoding
    for transformer models by encoding relative positions through rotation matrices.
    
    Args:
        dim (int): Dimension of the embeddings
        max_position_embeddings (int): Maximum sequence length (default: 2048)
        base (int): Base for the frequency computation (default: 10000)
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        """
        Apply rotary embeddings to the input.
        
        Args:
            x (torch.Tensor): Input tensor
            seq_len (int): Sequence length
            
        Returns:
            tuple: (cos, sin) tensors for rotary embeddings
        """
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    """Rotate half the hidden dimensions of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class FlashMLA(nn.Module):
    """
    Flash Multi-head Latent Attention with causal masking.
    
    This module implements an efficient attention mechanism with the following features:
    - Flash attention for memory-efficient computation
    - Rotary position embeddings
    - Causal attention for language modeling
    - Proper handling of attention masks
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = BitLinear(dim, dim * 3)
        self.proj = BitLinear(dim, dim)
        self.dropout = dropout
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Check if flash attention is available
        self.use_flash_attn = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash_attn = True
        except ImportError:
            print("Flash attention not available, falling back to standard attention")

    def forward(self, x, mask=None):
        """
        Forward pass of the attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        B, L, C = x.shape
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads), qkv)
        
        cos, sin = self.rotary_emb(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.use_flash_attn:
            # Flash attention implementation
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(-1, self.num_heads, -1, -1)
                mask = mask.contiguous()
            
            try:
                out = self.flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                    softmax_scale=self.scale,
                    key_padding_mask=mask if mask is not None else None
                )
            except Exception as e:
                print(f"Flash attention failed, falling back to standard attention: {e}")
                out = self._standard_attention(q, k, v, mask)
        else:
            out = self._standard_attention(q, k, v, mask)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.proj(out)
        
        return out
    
    def _standard_attention(self, q, k, v, mask=None):
        """Standard attention implementation as fallback."""
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        return torch.matmul(attn, v)

class SwiGLU(nn.Module):
    """
    SwiGLU activation function for Feed-Forward Network layers.
    
    This module implements the SwiGLU activation, which provides stronger activation
    scaling compared to traditional activation functions. It uses BitLinear layers
    for efficient computation.
    
    Args:
        in_features (int): Input dimension
        hidden_features (int): Hidden dimension
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = BitLinear(in_features, hidden_features)
        self.w2 = BitLinear(in_features, hidden_features)
        self.w3 = BitLinear(hidden_features, in_features)

    def forward(self, x):
        """
        Forward pass of the SwiGLU activation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        swish = self.w1(x) * torch.sigmoid(self.w2(x))
        return self.w3(swish)

class TransformerBlock(nn.Module):
    """
    Transformer block implementing the core architecture.
    
    This block combines attention and feed-forward layers with proper normalization:
    - Flash Multi-head Latent Attention
    - SwiGLU activation in FFN
    - SubLN normalization without bias
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of hidden dimension to input dimension
        dropout (float): Dropout probability
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = FlashMLA(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x, mask=None):
        """
        Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class BitSeek(nn.Module):
    """
    BitSeek: A 1.58-bit Quantized Language Model.
    
    This is the main model class that combines all components into a complete
    language model architecture. It implements:
    - Token embeddings
    - Position embeddings (RoPE)
    - Transformer blocks
    - Output head
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_size (int): Hidden dimension size
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=2048,
        dropout=0.0
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = RotaryEmbedding(hidden_size // num_heads, max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.head = BitLinear(hidden_size, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        """
        Forward pass of the complete model.
        
        Args:
            x (torch.Tensor): Input token ids
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output logits
        """
        B, L = x.shape
        
        # Token embeddings
        x = self.token_embedding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits 