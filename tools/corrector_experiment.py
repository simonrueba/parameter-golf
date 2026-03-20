#!/usr/bin/env python3
"""
Corrector experiment v2: Two-point gated repair on real FineWeb data.

Based on toy-proxy findings:
- Two-point gated CE-only (mid0 + final, r16+r16) is the champion
- Earlier repair placement is much better
- CE-only beats CE+KL distillation
- Gating helps a lot (selective correction)

Usage (on RunPod after training run):
    python3 tools/corrector_experiment.py --checkpoint final_model.pt
"""
from __future__ import annotations
import argparse, copy, glob, io, math, os, sys, time, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, RMSNorm, Block,
    load_data_shard, load_validation_tokens, build_sentencepiece_luts,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    INT8_KEEP_FLOAT_MAX_NUMEL, INT8_KEEP_FLOAT_STORE_DTYPE, INT8_CLIP_Q,
)

# ---------------------------------------------------------------------------
# Quantization at INT6 / INT4
# ---------------------------------------------------------------------------
def _quantize_tensor(t: Tensor, max_val: int):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -max_val - 1, max_val).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max_val if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val - 1, max_val).to(torch.int8)
    return q.contiguous(), scale

def _dequantize_tensor(q: Tensor, s: Tensor, dtype):
    if s.ndim > 0:
        return (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
    return (q.float() * float(s.item())).to(dtype)

def quantize_sd(state_dict: dict[str, Tensor], bits: int):
    if bits == 8:
        obj, stats = quantize_state_dict_int8(state_dict)
        buf = io.BytesIO(); torch.save(obj, buf)
        return obj, zlib.compress(buf.getvalue(), 9), stats.get("int8_payload_bytes", 0)
    max_val = {6: 31, 4: 7}[bits]
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    payload_bytes = 0
    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            pt = t.to(INT8_KEEP_FLOAT_STORE_DTYPE) if t.is_floating_point() and t.dtype in {torch.float32, torch.bfloat16} else t
            passthrough[name] = pt
            payload_bytes += pt.numel() * pt.element_size()
            continue
        q, s = _quantize_tensor(t, max_val)
        quantized[name] = q; scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        payload_bytes += q.numel() + s.numel() * 2
    obj = {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "bits": bits}
    buf = io.BytesIO(); torch.save(obj, buf)
    return obj, zlib.compress(buf.getvalue(), 9), payload_bytes

def dequantize_sd(obj, bits: int):
    if bits == 8:
        return dequantize_state_dict_int8(obj)
    out = {}
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        out[name] = _dequantize_tensor(q, obj["scales"][name], dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out[name] = t.detach().cpu().contiguous()
    return out

# ---------------------------------------------------------------------------
# Corrector modules (proven winners from toy-proxy experiments)
# ---------------------------------------------------------------------------
class AffineCorrector(nn.Module):
    """h = a * h + b. Cheapest possible: 2*D params."""
    def __init__(self, dim: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, h: Tensor) -> Tensor:
        return self.a * h + self.b

class LowRankCorrector(nn.Module):
    """h = h + up(down(h)). Plain low-rank: 2*D*r params."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self, h: Tensor) -> Tensor:
        return h + self.up(self.down(h))

class GatedLowRankCorrector(nn.Module):
    """h = h + gate(h) * up(down(h)). Selective correction: 3*D*r + r params."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.gate_proj = nn.Linear(dim, rank, bias=True)
        nn.init.zeros_(self.up.weight)
    def forward(self, h: Tensor) -> Tensor:
        delta = self.up(self.down(h))
        gate = torch.sigmoid(self.gate_proj(h))
        return h + gate * self.up(self.down(h))

# ---------------------------------------------------------------------------
# Two-point corrector: repairs at an intermediate block + final hidden state
# ---------------------------------------------------------------------------
class TwoPointCorrector(nn.Module):
    """Applies correction at block `mid_idx` output AND after final_norm."""
    def __init__(self, dim: int, rank_mid: int, rank_final: int, gated: bool = False):
        super().__init__()
        cls = GatedLowRankCorrector if gated else LowRankCorrector
        self.mid_corrector = cls(dim, rank_mid)
        self.final_corrector = cls(dim, rank_final)

    def param_bytes(self) -> int:
        return sum(p.numel() for p in self.parameters()) * 2  # FP16

# ---------------------------------------------------------------------------
# Forward pass with two-point correction injected
# ---------------------------------------------------------------------------
def forward_with_correction(
    raw_model, x: Tensor, corrector=None, mid_idx: int = 0,
) -> Tensor:
    """Run frozen backbone forward, injecting corrector at mid_idx and final.
    Backbone runs under no_grad; only corrector outputs carry gradients."""
    with torch.no_grad():
        x_emb = raw_model.tok_emb(x)
        x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))
        x0 = x_emb
        h = x_emb
        skips = []
        block_count = 0
        for i in range(raw_model.num_encoder_layers):
            h = raw_model.blocks[i](h, x0)
            skips.append(h)
            block_count += 1
            if corrector is not None and (block_count - 1) == mid_idx:
                break
        # Mid correction (with gradients)
    if corrector is not None and block_count - 1 == mid_idx:
        h = corrector.mid_corrector(h.detach().requires_grad_(True))
        skips[-1] = h
    # Continue backbone under no_grad
    with torch.no_grad():
        start_enc = block_count
        for i in range(start_enc, raw_model.num_encoder_layers):
            h = raw_model.blocks[i](h.detach(), x0)
            skips.append(h)
        for i in range(raw_model.num_decoder_layers):
            if skips:
                h = h + raw_model.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
            h = raw_model.blocks[raw_model.num_encoder_layers + i](h, x0)
        h = raw_model.final_norm(h)
    # Final correction (with gradients)
    if corrector is not None:
        h = corrector.final_corrector(h.detach().requires_grad_(True))
    return h

def get_logits(raw_model, h: Tensor) -> Tensor:
    h_flat = h.reshape(-1, h.size(-1))
    if raw_model.tie_embeddings:
        logits_proj = F.linear(h_flat, raw_model.tok_emb.weight)
    else:
        logits_proj = raw_model.lm_head(h_flat)
    return raw_model.logit_softcap * torch.tanh(logits_proj / raw_model.logit_softcap)

# ---------------------------------------------------------------------------
# Evaluate BPB (chunked, fast)
# ---------------------------------------------------------------------------
def evaluate_bpb(raw_model, val_tokens, seq_len, base_bytes, has_space, is_boundary,
                 device, corrector=None, mid_idx=0):
    total = val_tokens.numel() - 1
    usable = (total // seq_len) * seq_len
    loss_sum, tok_count, byte_count = 0.0, 0, 0.0
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, usable, seq_len * 8):
            end = min(start + seq_len * 8, usable)
            n_seqs = (end - start) // seq_len
            local = val_tokens[start:start + n_seqs * seq_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(n_seqs, seq_len)
            y = local[1:].reshape(n_seqs, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                h = forward_with_correction(raw_model, x, corrector, mid_idx)
                logits = get_logits(raw_model, h)
                batch_loss = F.cross_entropy(logits.float(), y.reshape(-1), reduction="mean")
            n_tok = y.numel()
            loss_sum += batch_loss.item() * n_tok
            tok_count += n_tok
            tb = base_bytes[y.reshape(-1)].to(torch.int16)
            tb += (has_space[y.reshape(-1)] & ~is_boundary[x.reshape(-1)]).to(torch.int16)
            byte_count += tb.float().sum().item()
    elapsed = (time.time() - t0) * 1000
    val_loss = loss_sum / tok_count
    bpb = (val_loss / math.log(2.0)) * (tok_count / byte_count)
    return val_loss, bpb, elapsed

# ---------------------------------------------------------------------------
# Train corrector (CE-only, proven better than CE+KL)
# ---------------------------------------------------------------------------
def train_corrector(raw_teacher, raw_student, corrector, val_tokens, device, args,
                    mid_idx=0, n_steps=300, lr=1e-3):
    seq_len = args.train_seq_len
    usable = ((val_tokens.numel() - 1) // seq_len) * seq_len
    optimizer = torch.optim.Adam(corrector.parameters(), lr=lr)
    corrector.train()
    losses = []
    for step in range(n_steps):
        starts = torch.randint(0, usable - seq_len, (4,))
        x = torch.stack([val_tokens[s:s + seq_len] for s in starts]).to(device=device, dtype=torch.int64)
        y = torch.stack([val_tokens[s + 1:s + seq_len + 1] for s in starts]).to(device=device, dtype=torch.int64)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = forward_with_correction(raw_student, x, corrector, mid_idx)
            logits = get_logits(raw_student, h)
            loss = F.cross_entropy(logits.float(), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == n_steps - 1:
            print(f"    step {step:>4}/{n_steps}  ce_loss: {loss.item():.4f}")
        losses.append(loss.item())
    return losses

# ---------------------------------------------------------------------------
# Build model from checkpoint
# ---------------------------------------------------------------------------
def build_model(args, device):
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for mod in m.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    return m

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final_model.pt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    cli = parser.parse_args()

    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = args.model_dim

    print("=" * 70)
    print("CORRECTOR EXPERIMENT v2: Two-point gated repair on FineWeb")
    print("=" * 70)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, args.vocab_size, device)
    print(f"Val tokens: {val_tokens.numel()-1:,}  model_dim: {D}  layers: {args.num_layers}")

    # Load teacher
    teacher = build_model(args, device)
    teacher.load_state_dict(torch.load(cli.checkpoint, map_location=device))
    teacher.eval()
    print(f"Loaded checkpoint: {cli.checkpoint}")

    # ── BASELINES ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINES: float / INT8 / INT6 / INT4")
    print("=" * 70)

    results = {}
    # Float
    vl, bpb, ms = evaluate_bpb(teacher, val_tokens, args.train_seq_len, base_bytes, has_space, is_boundary, device)
    results["float"] = {"bpb": bpb, "bytes": 0}
    print(f"\n  float    bpb: {bpb:.6f}  loss: {vl:.6f}  ({ms:.0f}ms)")

    # Quantized baselines
    int4_model = None
    for bits in [8, 6, 4]:
        sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
        obj, compressed, payload = quantize_sd(sd, bits)
        dq = dequantize_sd(obj, bits)
        qm = build_model(args, device)
        qm.load_state_dict(dq, strict=False)
        qm.eval()
        vl, bpb, ms = evaluate_bpb(qm, val_tokens, args.train_seq_len, base_bytes, has_space, is_boundary, device)
        results[f"int{bits}"] = {"bpb": bpb, "bytes": len(compressed), "gap": bpb - results["float"]["bpb"]}
        print(f"  INT{bits:<5} bpb: {bpb:.6f}  loss: {vl:.6f}  compressed: {len(compressed):>10,}  gap: {bpb - results['float']['bpb']:+.6f}  ({ms:.0f}ms)")
        if bits == 4:
            int4_model = qm

    int4_bpb = results["int4"]["bpb"]
    int4_gap = int4_bpb - results["float"]["bpb"]
    print(f"\n  INT4 gap to recover: {int4_gap:.6f} BPB")

    # ── CORRECTOR SWEEP ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CORRECTOR SWEEP (frozen INT4 backbone, CE-only)")
    print("=" * 70)

    # Clear inference-mode cached RoPE tensors before training with gradients
    for m in [teacher, int4_model]:
        for block in m.blocks:
            block.attn.rotary._cos_cached = None
            block.attn.rotary._sin_cached = None
            block.attn.rotary._seq_len_cached = 0

    n_blocks = args.num_layers
    mid_idx = 0  # earliest block — proven best placement

    experiments = [
        ("affine-final", None, 0),  # special: affine only at final
        ("plain-final-r16", "plain-final", 16),
        ("plain-2pt-mid0-r16+r16", "plain-2pt", 16),
        ("gated-2pt-mid0-r16+r16", "gated-2pt", 16),
        ("gated-2pt-mid0-r8+r16", "gated-2pt-8-16", 0),
    ]

    corrector_results = {}

    for label, mode, rank in experiments:
        if label == "affine-final":
            corr = AffineCorrector(D).to(device).float()
            # Wrap in TwoPointCorrector-like interface
            class _AffineWrap(nn.Module):
                def __init__(self, aff):
                    super().__init__()
                    self.mid_corrector = nn.Identity()
                    self.final_corrector = aff
            corrector = _AffineWrap(corr)
            n_params = 2 * D
        elif mode == "plain-final":
            corr = LowRankCorrector(D, rank).to(device).float()
            class _FinalWrap(nn.Module):
                def __init__(self, c):
                    super().__init__()
                    self.mid_corrector = nn.Identity()
                    self.final_corrector = c
            corrector = _FinalWrap(corr)
            n_params = 2 * D * rank
        elif mode == "plain-2pt":
            corrector = TwoPointCorrector(D, rank, rank, gated=False).to(device).float()
            n_params = sum(p.numel() for p in corrector.parameters())
        elif mode == "gated-2pt":
            corrector = TwoPointCorrector(D, rank, rank, gated=True).to(device).float()
            n_params = sum(p.numel() for p in corrector.parameters())
        elif mode == "gated-2pt-8-16":
            corrector = TwoPointCorrector(D, 8, 16, gated=True).to(device).float()
            n_params = sum(p.numel() for p in corrector.parameters())
        else:
            continue

        helper_bytes = n_params * 2  # FP16
        print(f"\n  --- {label} ({n_params:,} params, {helper_bytes:,} bytes FP16) ---")

        train_corrector(teacher, int4_model, corrector, val_tokens, device, args,
                        mid_idx=mid_idx, n_steps=cli.steps, lr=cli.lr)

        corrector.eval()
        vl, bpb, ms = evaluate_bpb(int4_model, val_tokens, args.train_seq_len,
                                    base_bytes, has_space, is_boundary, device,
                                    corrector=corrector, mid_idx=mid_idx)
        recovery = int4_bpb - bpb
        pct = recovery / int4_gap * 100 if int4_gap > 0 else 0
        total_bytes = results["int4"]["bytes"] + helper_bytes
        corrector_results[label] = {"bpb": bpb, "recovery": recovery, "helper_bytes": helper_bytes, "total_bytes": total_bytes}
        print(f"    bpb: {bpb:.6f}  recovery: {recovery:+.6f} ({pct:.1f}%)  total: {total_bytes:,} bytes  ({ms:.0f}ms)")

    # ── SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"{'Model':<30} {'Bytes':>12} {'BPB':>10} {'vs Float':>10} {'vs INT4':>10}")
    print("-" * 72)
    for name in ["float", "int8", "int6", "int4"]:
        r = results[name]
        gap = r["bpb"] - results["float"]["bpb"]
        print(f"{name:<30} {r.get('bytes', 0):>12,} {r['bpb']:>10.6f} {gap:>+10.6f} {'':>10}")
    for name, r in corrector_results.items():
        gap_f = r["bpb"] - results["float"]["bpb"]
        print(f"INT4+{name:<24} {r['total_bytes']:>12,} {r['bpb']:>10.6f} {gap_f:>+10.6f} {r['recovery']:>+10.6f}")

    print(f"\n  Key comparison: INT6 at {results['int6']['bytes']:,} bytes = {results['int6']['bpb']:.6f} BPB")
    best = min(corrector_results.values(), key=lambda x: x["bpb"])
    best_name = min(corrector_results, key=lambda k: corrector_results[k]["bpb"])
    print(f"  Best corrector ({best_name}) at {best['total_bytes']:,} bytes = {best['bpb']:.6f} BPB")
    if best["bpb"] < results["int6"]["bpb"]:
        print(f"  => CORRECTOR WINS by {results['int6']['bpb'] - best['bpb']:.6f} BPB")
    else:
        print(f"  => INT6 wins by {best['bpb'] - results['int6']['bpb']:.6f} BPB")


if __name__ == "__main__":
    main()
