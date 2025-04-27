# BitSeek: A 1.58-bit Quantized Language Model Implementation

## Overview
This PR adds the BitSeek implementation, a tiny quantized language model with the following features:
- 1.58-bit weight quantization (ternary values)
- 8-bit activation quantization
- Flash Multi-head Latent Attention with causal masking
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in FFN layers
- SubLN normalization
- No bias terms in linear or normalization layers

## Key Components
1. **BitLinear Layer**
   - Implements 1.58-bit weight quantization using ternary values {-1, 0, +1}
   - 8-bit activations quantized using absmax quantization (per-token)
   - No bias terms for efficiency

2. **FlashMLA (Flash Multi-head Latent Attention)**
   - Memory-efficient attention mechanism with automatic fallback
   - Rotary Position Embeddings
   - Causal attention for language modeling

3. **SwiGLU Activation**
   - Used in Feed-Forward Network layers
   - Provides stronger activation scaling
   - Implemented with BitLinear layers

## Performance
- ~300M parameters
- 2048 sequence length
- Efficient memory usage through quantization
- Fast inference with flash attention

## Testing
The implementation has been tested with:
- PyTorch 2.0.0+
- Flash Attention 2.0.0+
- Various sequence lengths up to 2048

## Dependencies
Added to requirements.txt:
- torch>=2.0.0
- einops>=0.6.0
- transformers>=4.30.0
- flash-attn>=2.0.0
- bitsandbytes>=0.41.0
- accelerate>=0.20.0
