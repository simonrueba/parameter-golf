#!/usr/bin/env python3
"""
Parameter budget calculator for the Parameter Golf challenge.

Computes unique parameter counts, estimated int8+zlib artifact sizes,
and explores the design space for shared-depth architectures.

Usage:
    python3 tools/param_budget.py
"""


def block_params(dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int) -> dict:
    """Parameter count for one transformer block."""
    head_dim = dim // num_heads
    kv_dim = num_kv_heads * head_dim

    attn_qkvo = dim * dim + 2 * dim * kv_dim + dim * dim  # Q, K, V, O projections
    attn_q_gain = num_heads  # per-head gain scalar
    mlp = dim * (mlp_mult * dim) + (mlp_mult * dim) * dim  # fc + proj
    scales = dim * 2  # attn_scale + mlp_scale
    resid_mix = dim * 2  # resid_mix (2, dim)

    return {
        "attn_qkvo": attn_qkvo,
        "attn_q_gain": attn_q_gain,
        "mlp": mlp,
        "scales": scales,
        "resid_mix": resid_mix,
        "total": attn_qkvo + attn_q_gain + mlp + scales + resid_mix,
    }


def model_params(
    vocab_size: int,
    dim: int,
    num_unique_layers: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    tie_embeddings: bool,
    num_skip_weights: int = 0,
) -> dict:
    """Total unique parameter count for a model configuration."""
    bp = block_params(dim, num_heads, num_kv_heads, mlp_mult)

    embedding = vocab_size * dim
    lm_head = 0 if tie_embeddings else vocab_size * dim
    skip = num_skip_weights * dim

    total = embedding + lm_head + num_unique_layers * bp["total"] + skip

    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "per_block": bp["total"],
        "all_blocks": num_unique_layers * bp["total"],
        "skip_weights": skip,
        "total": total,
    }


def estimate_int8_zlib_bytes(params: dict, code_bytes: int = 48000) -> dict:
    """
    Estimate compressed artifact size.

    Heuristics from the baseline:
    - 17,059,912 params -> 15,815,847 bytes int8+zlib (0.926 bytes/param)
    - Large matrices: int8 (1 byte/param) + fp16 row scales (~2/row)
    - Small tensors (<65536 elements): stored as fp16 (2 bytes/param)
    - zlib typically achieves ~0.90-0.95 ratio on int8 weights
    """
    large_matrix_params = params["all_blocks"] + params["lm_head"]
    small_params = params["embedding"] + params["skip_weights"]

    # int8 matrices: 1 byte/param + row scales (negligible)
    # fp16 small tensors: 2 bytes/param
    # zlib compression ratio ~0.92 on int8 weights
    raw_bytes = large_matrix_params * 1.0 + small_params * 2.0

    # Add torch.save overhead (~5KB) and zlib metadata
    zlib_ratio = 0.92  # conservative
    estimated_model_bytes = int(raw_bytes * zlib_ratio + 5000)
    estimated_total = estimated_model_bytes + code_bytes

    return {
        "model_bytes": estimated_model_bytes,
        "code_bytes": code_bytes,
        "total_bytes": estimated_total,
        "budget_remaining": 16_000_000 - estimated_total,
        "fits": estimated_total <= 16_000_000,
    }


def print_config(label, vocab, dim, unique_layers, effective_depth, heads, kv_heads, mlp_mult, tie_emb, skip_w=0):
    """Print a single configuration's budget analysis."""
    p = model_params(vocab, dim, unique_layers, heads, kv_heads, mlp_mult, tie_emb, skip_w)
    s = estimate_int8_zlib_bytes(p)

    marker = "OK" if s["fits"] else "OVER"
    print(f"  [{marker:>4}] {label}")
    print(f"         dim={dim} unique_layers={unique_layers} eff_depth={effective_depth} "
          f"heads={heads} kv={kv_heads} mlp={mlp_mult}x vocab={vocab} tie={tie_emb}")
    print(f"         params: {p['total']:>12,}  (blocks: {p['all_blocks']:>10,}  emb: {p['embedding']:>8,})")
    print(f"         est size: {s['total_bytes']:>12,} bytes  remaining: {s['budget_remaining']:>10,} bytes")
    print()


def main():
    print("=" * 80)
    print("PARAMETER GOLF - BUDGET CALCULATOR")
    print("=" * 80)
    print(f"Budget: 16,000,000 bytes (code + int8+zlib model)")
    print(f"Assuming code_bytes ~ 48,000")
    print()

    # =========================================================================
    print("-" * 80)
    print("BASELINE (for reference)")
    print("-" * 80)
    print_config("Baseline (actual: 15,863,489 bytes)",
                 1024, 512, 9, 9, 8, 4, 2, True, skip_w=4)

    # =========================================================================
    print("-" * 80)
    print("SHARED-DEPTH: Prelude(1) + Shared(1, looped N) + Coda(1) = 3 unique blocks")
    print("-" * 80)

    for dim in [640, 704, 768, 832, 896]:
        heads = max(4, dim // 64)
        kv_heads = max(2, heads // 2)
        for loops in [5, 7, 9]:
            eff = 2 + loops
            label = f"dim={dim} loops={loops} (eff_depth={eff})"
            print_config(label, 1024, dim, 3, eff, heads, kv_heads, 2, True, skip_w=0)

    # =========================================================================
    print("-" * 80)
    print("SHARED-DEPTH WITH LARGER VOCAB (sp4096)")
    print("-" * 80)

    for dim in [640, 704, 768]:
        heads = max(4, dim // 64)
        kv_heads = max(2, heads // 2)
        for vocab in [4096]:
            for loops in [5, 7]:
                eff = 2 + loops
                label = f"vocab={vocab} dim={dim} loops={loops}"
                print_config(label, vocab, dim, 3, eff, heads, kv_heads, 2, True, skip_w=0)

    # =========================================================================
    print()
    print("-" * 80)
    print("SWEET SPOT ANALYSIS")
    print("-" * 80)
    print("Finding configs that maximize dim x effective_depth while fitting in budget...")
    print()

    results = []
    for dim in range(512, 1025, 16):
        for unique in [2, 3, 4]:
            for loops in range(3, 15):
                heads = max(4, dim // 64)
                kv_heads = max(2, heads // 2)
                if dim % heads != 0:
                    continue
                if heads % kv_heads != 0:
                    continue
                eff = (unique - 1) + loops if unique >= 2 else loops
                p = model_params(1024, dim, unique, heads, kv_heads, 2, True, num_skip_weights=0)
                s = estimate_int8_zlib_bytes(p)
                if s["fits"]:
                    # Score: wider is better (depth delusion says width ~2.8x more important)
                    score = dim * 2.8 + eff * dim * 0.1
                    results.append((score, dim, unique, loops, eff, heads, kv_heads, p["total"], s["total_bytes"]))

    results.sort(reverse=True)
    print(f"{'Score':>8} {'Dim':>5} {'Uniq':>5} {'Loops':>5} {'Eff':>5} {'Heads':>5} "
          f"{'KV':>5} {'Params':>12} {'Size':>12}")
    for score, dim, unique, loops, eff, heads, kv_heads, params, size in results[:25]:
        print(f"{score:>8.0f} {dim:>5} {unique:>5} {loops:>5} {eff:>5} {heads:>5} "
              f"{kv_heads:>5} {params:>12,} {size:>12,}")


if __name__ == "__main__":
    main()
