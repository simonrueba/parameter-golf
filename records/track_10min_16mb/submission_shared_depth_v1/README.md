# Shared-Depth GPT (Prelude-Shared-Coda)

## Architecture

Replaces the baseline's 9 unique transformer blocks with a **3 unique block** shared-depth architecture:

- **Prelude** (1 block): processes raw embeddings once
- **Shared** (1 block, looped 7 times): weight-shared recurrent core
- **Coda** (1 block): final representation refinement

Width increased from 512 to **768** (12 heads, 6 KV heads, 2x MLP expansion).

This gives an effective depth of 9 (1 + 7 + 1) with only 3 unique parameter blocks, freeing parameter budget for a 50% wider model. The shared block receives sinusoidal depth embeddings (non-learnable, scaled by 0.1) to distinguish iterations, and x0 (post-embedding) is re-injected at each iteration via the existing resid_mix mechanism.

## Evaluation

Uses **strided sliding-window evaluation** (stride=512, window=1024) for the final BPB score. Each token is scored with full left context rather than position-dependent context from non-overlapping chunks. Both strided and standard chunked eval are reported for comparability.

## Key Design Decisions

1. **Width over unique depth**: The Depth Delusion paper suggests width matters ~2.8x more than depth at small scale. 768 vs 512 is the main capacity gain.
2. **7 shared iterations**: Local experiments showed 7 loops optimal; 5 too shallow, 9 shows gradient degradation.
3. **No U-Net skips**: Simplified to sequential with input injection (x0 re-injection) since skip connections don't compose cleanly with shared-weight loops.
4. **Parameter budget**: ~13.2M unique params, estimated ~12.3MB after int8+zlib, leaving ~3.7MB headroom under the 16MB cap.

## Configuration

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key env overrides (all have sensible defaults):
- `SHARED_ITERS=7` (shared block loop count)
- `MODEL_DIM=768` (width)
- `NUM_HEADS=12`, `NUM_KV_HEADS=6`
- `NUM_LAYERS=3` (unique blocks)
- `EVAL_STRIDE=512` (strided eval stride)

## Key Metrics

<!-- Fill after first RunPod run -->
- Pre-quant val_bpb: TBD
- Post-quant val_bpb: TBD
- Compressed artifact size: TBD
- Training steps completed: TBD
- Train time: TBD
