#!/usr/bin/env python3
"""
Head-to-head training comparison: baseline architecture vs shared-depth.
Runs on MLX with Adam (skipping Muon for speed) on real FineWeb data.

Compares training loss curves to validate shared-depth is competitive.

Usage:
    python3 tools/experiment_compare.py
"""
from __future__ import annotations

import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ---------------------------------------------------------------------------
# Data loading (from train_gpt_mlx.py)
# ---------------------------------------------------------------------------

COMPUTE_DTYPE = mx.bfloat16
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
SEQ_LEN = 256  # short for speed


def load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    n_tokens = int(header[2])
    return np.fromfile(path, dtype="<u2", count=n_tokens, offset=256 * 4)


def load_train_tokens():
    files = sorted(glob.glob(f"{DATA_PATH}/fineweb_train_*.bin"))
    if not files:
        raise FileNotFoundError(f"No train files in {DATA_PATH}")
    return load_shard(Path(files[0]))


def load_val_tokens():
    files = sorted(glob.glob(f"{DATA_PATH}/fineweb_val_*.bin"))
    if not files:
        raise FileNotFoundError(f"No val files in {DATA_PATH}")
    return load_shard(Path(files[0]))


def get_batch(tokens: np.ndarray, batch_size: int, seq_len: int):
    max_start = len(tokens) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([tokens[s:s + seq_len] for s in starts])
    y = np.stack([tokens[s + 1:s + seq_len + 1] for s in starts])
    return mx.array(x.astype(np.int32)), mx.array(y.astype(np.int32))


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        scale = 1.0 / math.sqrt(in_dim)
        self.weight = mx.random.uniform(-scale, scale, (out_dim, in_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * 1.5
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int = 2):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mlp_mult)
        self.proj = CastedLinear(dim * mlp_mult, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads)
        self.mlp = MLP(dim)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack([np.ones(dim, dtype=np.float32),
                                            np.zeros(dim, dtype=np.float32)]))
        self.attn.proj.weight = mx.zeros_like(self.attn.proj.weight)
        self.mlp.proj.weight = mx.zeros_like(self.mlp.proj.weight)

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(rms_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(rms_norm(x))
        return x


# ---------------------------------------------------------------------------
# Model A: Baseline (N unique layers)
# ---------------------------------------------------------------------------

class BaselineGPT(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int, heads: int, kv_heads: int):
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab, dim)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape) * 0.005).astype(COMPUTE_DTYPE)
        self.blocks = [Block(dim, heads, kv_heads) for _ in range(n_layers)]
        n_enc = n_layers // 2
        n_dec = n_layers - n_enc
        self.skip_weights = mx.ones((min(n_enc, n_dec), dim), dtype=mx.float32)
        self.n_enc = n_enc

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips = []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(len(self.blocks) - self.n_enc):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
        return rms_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.dim)
        y = target_ids.reshape(-1)
        logits = 30.0 * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / 30.0)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ---------------------------------------------------------------------------
# Model B: Shared-depth (prelude + shared×N + coda)
# ---------------------------------------------------------------------------

class SharedDepthGPT(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, kv_heads: int, shared_iters: int):
        super().__init__()
        self.dim = dim
        self.shared_iters = shared_iters
        self.tok_emb = nn.Embedding(vocab, dim)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape) * 0.005).astype(COMPUTE_DTYPE)
        self.prelude = Block(dim, heads, kv_heads)
        self.shared = Block(dim, heads, kv_heads)
        self.coda = Block(dim, heads, kv_heads)
        # Sinusoidal depth embeddings (not learnable, stored as buffer)
        embs = np.zeros((shared_iters, dim), dtype=np.float32)
        pos = np.arange(shared_iters)[:, None]
        div = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        embs[:, 0::2] = np.sin(pos * div)
        embs[:, 1::2] = np.cos(pos * div[:dim // 2])
        self._depth_embs = mx.array(embs * 0.1)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        x = self.prelude(x, x0)
        for i in range(self.shared_iters):
            x = self.shared(x + self._depth_embs[i].astype(x.dtype)[None, None, :], x0)
        x = self.coda(x, x0)
        return rms_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.dim)
        y = target_ids.reshape(-1)
        logits = 30.0 * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / 30.0)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def count_params(model) -> int:
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def train_and_eval(label: str, model, train_tokens: np.ndarray, n_steps: int = 100,
                   batch_size: int = 4, lr: float = 3e-4) -> list[float]:
    n_params = count_params(model)
    print(f"\n{'─' * 60}")
    print(f"{label}")
    print(f"  params: {n_params:,}  est_size: {int(n_params * 0.926 + 48000):,} bytes")
    print(f"  steps: {n_steps}  batch: {batch_size}  seq_len: {SEQ_LEN}  lr: {lr}")
    print(f"{'─' * 60}")

    loss_fn = nn.value_and_grad(model, model.loss)
    optimizer = optim.Adam(learning_rate=lr)

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        x, y = get_batch(train_tokens, batch_size, SEQ_LEN)
        loss_val, grads = loss_fn(x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        l = loss_val.item()
        losses.append(l)
        if step < 5 or (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            ms_per_step = elapsed / (step + 1) * 1000
            print(f"  step {step + 1:>4}/{n_steps}  loss: {l:.4f}  "
                  f"({ms_per_step:.0f} ms/step)")

    elapsed = time.time() - t0
    print(f"  total: {elapsed:.1f}s  final_loss: {losses[-1]:.4f}  "
          f"best_loss: {min(losses):.4f}")
    return losses


def main():
    print("=" * 60)
    print("PARAMETER GOLF - ARCHITECTURE COMPARISON")
    print("=" * 60)

    np.random.seed(42)
    train_tokens = load_train_tokens()
    print(f"Loaded {len(train_tokens):,} training tokens")

    N_STEPS = 100
    BATCH = 4
    LR = 3e-4
    VOCAB = 1024

    # ── Config A: Baseline (9 unique layers, dim=512) ─────────────
    mx.random.seed(42)
    model_a = BaselineGPT(vocab=VOCAB, dim=512, n_layers=9, heads=8, kv_heads=4)
    losses_a = train_and_eval(
        "A: Baseline (9 unique, dim=512, 17M params)",
        model_a, train_tokens, N_STEPS, BATCH, LR,
    )

    # ── Config B: Shared-depth (dim=768, 3 unique, 7 loops) ──────
    mx.random.seed(42)
    model_b = SharedDepthGPT(vocab=VOCAB, dim=768, heads=12, kv_heads=6, shared_iters=7)
    losses_b = train_and_eval(
        "B: Shared-depth (3 unique, 7 loops, dim=768, 13M params)",
        model_b, train_tokens, N_STEPS, BATCH, LR,
    )

    # ── Config C: Shared-depth (dim=768, 3 unique, 5 loops) ──────
    mx.random.seed(42)
    model_c = SharedDepthGPT(vocab=VOCAB, dim=768, heads=12, kv_heads=6, shared_iters=5)
    losses_c = train_and_eval(
        "C: Shared-depth (3 unique, 5 loops, dim=768, 13M params)",
        model_c, train_tokens, N_STEPS, BATCH, LR,
    )

    # ── Config D: Shared-depth (dim=768, 3 unique, 9 loops) ──────
    mx.random.seed(42)
    model_d = SharedDepthGPT(vocab=VOCAB, dim=768, heads=12, kv_heads=6, shared_iters=9)
    losses_d = train_and_eval(
        "D: Shared-depth (3 unique, 9 loops, dim=768, 13M params)",
        model_d, train_tokens, N_STEPS, BATCH, LR,
    )

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY (loss at step 100)")
    print(f"{'=' * 60}")
    results = [
        ("A: Baseline 9×512", losses_a[-1], min(losses_a), count_params(model_a)),
        ("B: Shared 7×768", losses_b[-1], min(losses_b), count_params(model_b)),
        ("C: Shared 5×768", losses_c[-1], min(losses_c), count_params(model_c)),
        ("D: Shared 9×768", losses_d[-1], min(losses_d), count_params(model_d)),
    ]
    print(f"{'Config':<25} {'Final':>8} {'Best':>8} {'Params':>12}")
    for label, final, best, params in results:
        print(f"{label:<25} {final:>8.4f} {best:>8.4f} {params:>12,}")

    # Loss trajectory comparison at key points
    print(f"\n{'Step':<6}", end="")
    for label, _, _, _ in results:
        print(f" {label:<15}", end="")
    print()
    for step in [0, 9, 19, 49, 99]:
        print(f"{step + 1:<6}", end="")
        for i, (_, _, _, _) in enumerate(results):
            all_losses = [losses_a, losses_b, losses_c, losses_d]
            print(f" {all_losses[i][step]:<15.4f}", end="")
        print()


if __name__ == "__main__":
    main()
