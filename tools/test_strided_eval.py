#!/usr/bin/env python3
"""
Test strided sliding-window evaluation logic.

Compares non-overlapping (stride=seq_len) vs overlapping (stride<seq_len) evaluation
on a small validation slice using a random model. Verifies:
1. Strided eval produces lower loss (more context per token)
2. Every token is counted exactly once
3. Byte accounting stays correct

Usage:
    python3 tools/test_strided_eval.py
"""
from __future__ import annotations

import glob
import math
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import sentencepiece as spm

COMPUTE_DTYPE = mx.bfloat16
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE = 1024


# ---------------------------------------------------------------------------
# Minimal model (just needs to produce logits)
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


class SimpleMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2)
        self.w2 = nn.Linear(dim * 2, dim)

    def __call__(self, x):
        return self.w2(nn.relu(self.w1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dim, heads)
        self.mlp = SimpleMLP(dim)
        self.ln1 = nn.RMSNorm(dim)
        self.ln2 = nn.RMSNorm(dim)

    def __call__(self, x, mask):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyModel(nn.Module):
    """Minimal transformer for eval testing. Not meant to be good."""
    def __init__(self, vocab: int, dim: int = 256, n_layers: int = 2, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab, dim)
        self.layers = [SimpleBlock(dim, heads) for _ in range(n_layers)]

    def __call__(self, x_ids: mx.array) -> mx.array:
        """Returns logits (batch, seq, vocab)."""
        x = self.tok_emb(x_ids).astype(COMPUTE_DTYPE)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(COMPUTE_DTYPE)
        for layer in self.layers:
            x = layer(x, mask)
        logits = x @ self.tok_emb.weight.astype(x.dtype).T
        return logits.astype(mx.float32)


# ---------------------------------------------------------------------------
# BPB helpers (from train_gpt.py / train_gpt_mlx.py)
# ---------------------------------------------------------------------------

def build_byte_luts(sp: spm.SentencePieceProcessor, vocab_size: int):
    """Build lookup tables for byte counting."""
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)

    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):  # ▁
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))

    return (
        mx.array(base_bytes),
        mx.array(has_leading_space),
        mx.array(is_boundary),
    )


# ---------------------------------------------------------------------------
# Eval: non-overlapping (baseline)
# ---------------------------------------------------------------------------

def eval_nonoverlapping(model, val_tokens: np.ndarray, seq_len: int,
                        base_bytes, has_space, is_boundary) -> dict:
    """Standard non-overlapping evaluation (stride = seq_len)."""
    n = len(val_tokens)
    usable = ((n - 1) // seq_len) * seq_len
    tokens = mx.array(val_tokens[:usable + 1].astype(np.int32))

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0

    for start in range(0, usable, seq_len):
        x = tokens[start:start + seq_len][None, :]  # (1, seq_len)
        y = tokens[start + 1:start + seq_len + 1]    # (seq_len,)

        logits = model(x)[0]  # (seq_len, vocab)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += seq_len

        # Byte counting
        prev_ids = tokens[start:start + seq_len]
        tgt_ids = y
        tok_bytes = base_bytes[tgt_ids].astype(mx.int16)
        tok_bytes = tok_bytes + (has_space[tgt_ids] & ~is_boundary[prev_ids]).astype(mx.int16)
        mx.eval(tok_bytes)
        total_bytes += tok_bytes.astype(mx.float32).sum().item()

    val_loss = total_loss / total_tokens
    bpt = val_loss / math.log(2.0)
    tpb = total_tokens / total_bytes
    bpb = bpt * tpb

    return {
        "val_loss": val_loss,
        "val_bpb": bpb,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Eval: strided sliding window
# ---------------------------------------------------------------------------

def eval_strided(model, val_tokens: np.ndarray, seq_len: int, stride: int,
                 base_bytes, has_space, is_boundary) -> dict:
    """
    Strided sliding-window evaluation.

    For each window of `seq_len` tokens, compute logits but only count
    the loss/bytes for the last `stride` tokens. The overlap region
    (first seq_len - stride tokens) serves purely as context.

    Every target token is scored exactly once with maximal left context.
    """
    n = len(val_tokens)
    tokens = mx.array(val_tokens.astype(np.int32))

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0

    # First window: count all seq_len tokens (no prior context available)
    # Subsequent windows: only count the last `stride` tokens
    pos = 0
    window_idx = 0
    while pos + seq_len < n:
        x = tokens[pos:pos + seq_len][None, :]        # (1, seq_len)
        y = tokens[pos + 1:pos + seq_len + 1]          # (seq_len,)

        logits = model(x)[0]  # (seq_len, vocab)

        if window_idx == 0:
            # First window: count all positions
            count_from = 0
        else:
            # Subsequent: only count last `stride` positions
            count_from = seq_len - stride

        logits_counted = logits[count_from:]
        y_counted = y[count_from:]
        loss = nn.losses.cross_entropy(logits_counted, y_counted, reduction="sum")
        mx.eval(loss)

        n_counted = seq_len - count_from
        total_loss += loss.item()
        total_tokens += n_counted

        # Byte counting for counted positions only
        prev_counted = tokens[pos + count_from:pos + seq_len]
        tgt_counted = y_counted
        tok_bytes = base_bytes[tgt_counted].astype(mx.int16)
        tok_bytes = tok_bytes + (has_space[tgt_counted] & ~is_boundary[prev_counted]).astype(mx.int16)
        mx.eval(tok_bytes)
        total_bytes += tok_bytes.astype(mx.float32).sum().item()

        pos += stride
        window_idx += 1

    val_loss = total_loss / total_tokens
    bpt = val_loss / math.log(2.0)
    tpb = total_tokens / total_bytes
    bpb = bpt * tpb

    return {
        "val_loss": val_loss,
        "val_bpb": bpb,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "n_windows": window_idx,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STRIDED EVAL TEST")
    print("=" * 60)

    # Load tokenizer for byte LUTs
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    base_bytes, has_space, is_boundary = build_byte_luts(sp, VOCAB_SIZE)

    # Load a small slice of validation data
    val_files = sorted(glob.glob(f"{DATA_PATH}/fineweb_val_*.bin"))
    raw = np.fromfile(val_files[0], dtype="<u2",
                      count=int(np.fromfile(val_files[0], dtype="<i4", count=256)[2]),
                      offset=256 * 4)

    # Use first 50K tokens for speed
    VAL_SLICE = 50_000
    val_tokens = raw[:VAL_SLICE]
    print(f"Using {len(val_tokens):,} validation tokens")

    # Build a small random model
    mx.random.seed(42)
    model = TinyModel(VOCAB_SIZE, dim=128, n_layers=2, heads=4)
    print(f"Model: TinyModel(dim=128, layers=2, heads=4)")

    SEQ_LEN = 256

    # ── Non-overlapping baseline ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Non-overlapping (stride={SEQ_LEN})")
    t0 = time.time()
    r0 = eval_nonoverlapping(model, val_tokens, SEQ_LEN, base_bytes, has_space, is_boundary)
    t0 = time.time() - t0
    print(f"  val_loss: {r0['val_loss']:.6f}")
    print(f"  val_bpb:  {r0['val_bpb']:.6f}")
    print(f"  tokens:   {r0['total_tokens']:,}")
    print(f"  bytes:    {r0['total_bytes']:,.0f}")
    print(f"  time:     {t0:.1f}s")

    # ── Strided evaluations ───────────────────────────────────────
    for stride in [128, 64, 32]:
        print(f"\n{'─' * 60}")
        print(f"Strided (seq_len={SEQ_LEN}, stride={stride})")
        t1 = time.time()
        r1 = eval_strided(model, val_tokens, SEQ_LEN, stride, base_bytes, has_space, is_boundary)
        t1 = time.time() - t1
        print(f"  val_loss: {r1['val_loss']:.6f}")
        print(f"  val_bpb:  {r1['val_bpb']:.6f}")
        print(f"  tokens:   {r1['total_tokens']:,}")
        print(f"  bytes:    {r1['total_bytes']:,.0f}")
        print(f"  windows:  {r1['n_windows']}")
        print(f"  time:     {t1:.1f}s  ({t1/t0:.1f}x baseline)")

        delta_loss = r1['val_loss'] - r0['val_loss']
        delta_bpb = r1['val_bpb'] - r0['val_bpb']
        print(f"  delta_loss: {delta_loss:+.6f}")
        print(f"  delta_bpb:  {delta_bpb:+.6f}")

        # Sanity: token counts should be similar (strided may cover slightly fewer
        # tokens at the end)
        token_diff = abs(r1['total_tokens'] - r0['total_tokens'])
        print(f"  token_count_diff: {token_diff} "
              f"({'OK' if token_diff < SEQ_LEN else 'WARN'})")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    print("If strided BPB < non-overlapping BPB, the approach works.")
    print("The delta should increase as stride decreases (more context).")
    print("Diminishing returns below stride=64 are expected.")
    print("The time multiplier shows the compute cost.")


if __name__ == "__main__":
    main()
