# blueberry-llm-rope
Research on RoPE on a small LLM

## Bugfix: RoPE transpose (heads vs sequence)

- **Issue**: `RotaryPositionalEmbeddings` expects inputs shaped `[B, T, H, D]`, but Q/K were `[B, H, T, D]`. Applying RoPE without transposing made it treat heads as the sequence dimension, breaking positional encoding.
- **Fix**: Transpose Q/K to `[B, T, H, D]` before RoPE, then transpose back to `[B, H, T, D]`.
- **Impact** (quick run):
  - Before (buggy): Val Loss 0.4204, Acc 0.9028, PPL 1.52
  - After (fixed): Val Loss 0.1365, Acc 0.9766, PPL 1.15

