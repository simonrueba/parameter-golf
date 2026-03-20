#!/usr/bin/env python3
"""
Parameter budget calculator v2 — calibrated with ACTUAL compression data.

From the 1xH100 run:
  - 13,181,988 unique params (dim=768, 3 unique blocks)
  - int8+zlib model: 9,495,190 bytes
  - code: 58,322 bytes
  - total: 9,553,512 bytes
  - actual ratio: 0.720 bytes/param (model_bytes / params)

The old estimate assumed 0.926 bytes/param — way too conservative.
The shared-depth model compresses much better because:
  - Only 3 unique blocks (fewer unique weight matrices)
  - zlib loves repeated patterns in int8 weights
  - Embedding table (768K params) stored as fp16 gets good compression

Usage:
    python3 tools/param_budget_v2.py
"""


def block_params(dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int) -> int:
    head_dim = dim // num_heads
    kv_dim = num_kv_heads * head_dim
    attn = dim * dim + 2 * dim * kv_dim + dim * dim  # Q, K, V, O
    mlp = 2 * dim * (mlp_mult * dim)  # fc + proj
    scalars = num_heads + dim * 2 + dim * 2  # q_gain + attn/mlp_scale + resid_mix
    return attn + mlp + scalars


def total_params(vocab: int, dim: int, unique_layers: int, heads: int, kv: int, mlp: int, tie: bool) -> int:
    emb = vocab * dim
    lm_head = 0 if tie else vocab * dim
    blocks = unique_layers * block_params(dim, heads, kv, mlp)
    return emb + lm_head + blocks


def estimate_bytes(params: int, code_bytes: int = 60000) -> int:
    """
    Calibrated estimate from actual run.

    Actual data point: 13,181,988 params -> 9,495,190 bytes model
    Ratio: 0.720 bytes/param

    But this ratio varies with model structure:
    - More unique layers -> slightly worse compression (more entropy)
    - Wider model -> similar ratio
    - Larger vocab -> worse ratio (embedding stored as fp16 = 2 bytes/param)

    Use 0.72 for the block params and 1.5 for embeddings (fp16 + zlib).
    """
    # Rough split: blocks vs embedding
    # For tied embeddings with vocab=1024: embedding is a small fraction
    # The 0.72 ratio already includes the embedding overhead for the calibration point
    model_bytes = int(params * 0.72)
    return model_bytes + code_bytes


def main():
    CODE = 60000  # slightly larger for v2 submission with more features

    print("=" * 85)
    print("PARAMETER GOLF - BUDGET CALCULATOR v2 (calibrated from actual run)")
    print("=" * 85)
    print(f"Budget: 16,000,000 bytes | Code ~{CODE:,} bytes")
    print(f"Calibration: 13.2M params -> 9.50MB model (0.720 bytes/param)")
    print(f"Max model bytes: {16_000_000 - CODE:,}")
    print()

    # Reference points
    print("-" * 85)
    print("REFERENCE POINTS")
    print("-" * 85)
    p = total_params(1024, 768, 3, 12, 6, 2, True)
    print(f"  Current (dim=768, 3 uniq):  {p:>12,} params  est: {estimate_bytes(p, CODE):>12,} bytes  "
          f"actual: 9,553,512 bytes")
    p = total_params(1024, 512, 9, 8, 4, 2, True)
    print(f"  Baseline (dim=512, 9 uniq): {p:>12,} params  actual: 15,863,489 bytes")
    print()

    # Sweep: find the widest model that fits
    print("-" * 85)
    print("SHARED-DEPTH SWEEP: 3 unique blocks, vocab=1024, tied embeddings")
    print("-" * 85)
    print(f"{'Dim':>5} {'Heads':>5} {'KV':>4} {'HD':>4} {'Params':>12} {'Est Bytes':>12} "
          f"{'Remaining':>10} {'Fits':>5}  {'Loops for eff=9':>15}")
    print("-" * 85)

    candidates = []
    for dim in range(768, 1281, 16):
        # Find valid head configs
        for hd in [48, 64, 80, 96, 128]:
            if dim % hd != 0:
                continue
            heads = dim // hd
            if heads < 2:
                continue
            for kv in [2, 3, 4, 6, 8]:
                if heads % kv != 0:
                    continue
                p = total_params(1024, dim, 3, heads, kv, 2, True)
                est = estimate_bytes(p, CODE)
                remaining = 16_000_000 - est
                fits = remaining >= 0
                if fits:
                    candidates.append((dim, heads, kv, hd, p, est, remaining))
                    print(f"{dim:>5} {heads:>5} {kv:>4} {hd:>4} {p:>12,} {est:>12,} "
                          f"{remaining:>10,} {'OK':>5}  loops=7 -> eff={2+7}")

    print()
    print("-" * 85)
    print("TOP CANDIDATES (widest that fit)")
    print("-" * 85)

    # Sort by dim (widest first), then by fewest heads (bigger head_dim = more expressive)
    candidates.sort(key=lambda x: (-x[0], x[3]))
    seen_dims = set()
    top = []
    for dim, heads, kv, hd, p, est, remaining in candidates:
        if dim not in seen_dims:
            seen_dims.add(dim)
            top.append((dim, heads, kv, hd, p, est, remaining))

    print(f"{'Dim':>5} {'Heads':>5} {'KV':>4} {'HD':>4} {'Params':>12} {'Est Bytes':>12} "
          f"{'Remaining':>10}")
    for dim, heads, kv, hd, p, est, remaining in top[-10:]:  # last 10 = widest
        marker = " <-- SWEET SPOT" if 500_000 < remaining < 2_000_000 else ""
        print(f"{dim:>5} {heads:>5} {kv:>4} {hd:>4} {p:>12,} {est:>12,} "
              f"{remaining:>10,}{marker}")

    # Also check 2 unique blocks (even wider)
    print()
    print("-" * 85)
    print("2 UNIQUE BLOCKS (prelude + shared only, no coda)")
    print("-" * 85)
    print(f"{'Dim':>5} {'Heads':>5} {'KV':>4} {'HD':>4} {'Params':>12} {'Est Bytes':>12} "
          f"{'Remaining':>10}")
    for dim in range(768, 1537, 16):
        for hd in [48, 64, 80, 96, 128]:
            if dim % hd != 0:
                continue
            heads = dim // hd
            if heads < 2:
                continue
            for kv in [2, 4, 6, 8]:
                if heads % kv != 0:
                    continue
                p = total_params(1024, dim, 2, heads, kv, 2, True)
                est = estimate_bytes(p, CODE)
                remaining = 16_000_000 - est
                if 0 < remaining < 2_000_000:
                    print(f"{dim:>5} {heads:>5} {kv:>4} {hd:>4} {p:>12,} {est:>12,} "
                          f"{remaining:>10,}")

    # Check with larger vocab
    print()
    print("-" * 85)
    print("LARGER VOCAB (sp4096) - 3 unique blocks")
    print("-" * 85)
    for dim in range(768, 1153, 16):
        for hd in [48, 64, 80, 96]:
            if dim % hd != 0:
                continue
            heads = dim // hd
            if heads < 2:
                continue
            for kv in [2, 4, 6, 8]:
                if heads % kv != 0:
                    continue
                p = total_params(4096, dim, 3, heads, kv, 2, True)
                est = estimate_bytes(p, CODE)
                remaining = 16_000_000 - est
                if 0 < remaining < 2_000_000:
                    print(f"  vocab=4096 dim={dim:>5} heads={heads:>3} kv={kv} hd={hd:>3}  "
                          f"params={p:>12,}  est={est:>12,}  rem={remaining:>10,}")


if __name__ == "__main__":
    main()
