#!/usr/bin/env python3
"""
Corrector experiment: Is INT4 error cheaply repairable?

Loads a trained model checkpoint, quantizes to INT8/INT6/INT4,
measures BPB for each, then trains tiny correctors on frozen INT4
backbone with teacher distillation.

Usage (on RunPod, after a training run that produced final_model.pt):
    python3 tools/corrector_experiment.py --checkpoint final_model.pt

Requires the training script's modules, so we import from the baseline.
"""
from __future__ import annotations
import argparse, copy, glob, io, math, os, sys, time, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Import model + data loading from the baseline script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, RMSNorm, Block,
    load_data_shard, load_validation_tokens,
    build_sentencepiece_luts,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL, INT8_KEEP_FLOAT_STORE_DTYPE,
    INT8_CLIP_Q,
)

# ---------------------------------------------------------------------------
# INT6 quantization (per-row, values in [-32, 31])
# ---------------------------------------------------------------------------
def quantize_int6_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -32, 31).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -32, 31).to(torch.int8)
    return q.contiguous(), scale

def dequantize_int6_tensor(q: Tensor, s: Tensor, dtype) -> Tensor:
    if s.ndim > 0:
        return (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
    return (q.float() * float(s.item())).to(dtype)

# ---------------------------------------------------------------------------
# INT4 quantization (per-row, values in [-8, 7])
# ---------------------------------------------------------------------------
def quantize_int4_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        scale = (clip_abs / 7.0).clamp_min(1.0 / 7.0)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -8, 7).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 7.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -8, 7).to(torch.int8)
    return q.contiguous(), scale

def dequantize_int4_tensor(q: Tensor, s: Tensor, dtype) -> Tensor:
    if s.ndim > 0:
        return (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
    return (q.float() * float(s.item())).to(dtype)

# ---------------------------------------------------------------------------
# Generic quantize/dequantize state dict at arbitrary bit width
# ---------------------------------------------------------------------------
def quantize_sd(state_dict: dict[str, Tensor], bits: int):
    """Quantize a state dict. Returns (quantized_sd, compressed_bytes)."""
    qfn = {8: None, 6: quantize_int6_tensor, 4: quantize_int4_tensor}[bits]
    dqfn = {8: None, 6: dequantize_int6_tensor, 4: dequantize_int4_tensor}[bits]

    if bits == 8:
        obj, stats = quantize_state_dict_int8(state_dict)
        buf = io.BytesIO(); torch.save(obj, buf)
        compressed = zlib.compress(buf.getvalue(), 9)
        return obj, compressed, stats.get("int8_payload_bytes", 0)

    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    total_bytes = 0
    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            pt = t.to(INT8_KEEP_FLOAT_STORE_DTYPE) if t.is_floating_point() and t.dtype in {torch.float32, torch.bfloat16} else t
            passthrough[name] = pt
            total_bytes += pt.numel() * pt.element_size()
            continue
        q, s = qfn(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        total_bytes += q.numel() * q.element_size() + s.numel() * s.element_size()

    obj = {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "bits": bits}
    buf = io.BytesIO(); torch.save(obj, buf)
    compressed = zlib.compress(buf.getvalue(), 9)
    return obj, compressed, total_bytes

def dequantize_sd(obj, bits: int) -> dict[str, Tensor]:
    if bits == 8:
        return dequantize_state_dict_int8(obj)
    dqfn = {6: dequantize_int6_tensor, 4: dequantize_int4_tensor}[bits]
    out = {}
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        out[name] = dqfn(q, obj["scales"][name], dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out[name] = t.detach().cpu().contiguous()
        if out[name].dtype == torch.float16 and name not in obj.get("dtypes", {}):
            # Might need to restore to original dtype — keep as-is for simplicity
            pass
    return out

# ---------------------------------------------------------------------------
# Corrector modules
# ---------------------------------------------------------------------------
class AffineCorrector(nn.Module):
    """h_final = a * h + b. 2*D params."""
    def __init__(self, dim: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, h: Tensor) -> Tensor:
        return self.a * h + self.b

class LowRankCorrector(nn.Module):
    """h_final = h + up(down(h)). 2*D*r params."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self, h: Tensor) -> Tensor:
        return h + self.up(self.down(h))

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_bpb(model: nn.Module, val_tokens: Tensor, seq_len: int,
                 base_bytes: Tensor, has_space: Tensor, is_boundary: Tensor,
                 device: torch.device, corrector: nn.Module | None = None) -> tuple[float, float, float]:
    """Evaluate chunked BPB. Returns (val_loss, val_bpb, eval_time_ms)."""
    total = val_tokens.numel() - 1
    usable = (total // seq_len) * seq_len
    loss_sum, tok_count, byte_count = 0.0, 0, 0.0
    model.eval()
    if corrector is not None:
        corrector.eval()

    # Unwrap for corrector injection
    raw = model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod

    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, usable, seq_len * 8):  # batch of 8 sequences
            end = min(start + seq_len * 8, usable)
            n_seqs = (end - start) // seq_len
            local = val_tokens[start:start + n_seqs * seq_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(n_seqs, seq_len)
            y = local[1:].reshape(n_seqs, seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if corrector is not None:
                    # Manual forward with corrector injection
                    x_emb = raw.tok_emb(x)
                    x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))
                    x0 = x_emb
                    h = x_emb
                    skips = []
                    for i in range(raw.num_encoder_layers):
                        h = raw.blocks[i](h, x0)
                        skips.append(h)
                    for i in range(raw.num_decoder_layers):
                        if skips:
                            h = h + raw.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
                        h = raw.blocks[raw.num_encoder_layers + i](h, x0)
                    h = raw.final_norm(h)
                    h = corrector(h)  # <-- corrector applied here
                    h = h.reshape(-1, h.size(-1))
                    if raw.tie_embeddings:
                        logits_proj = F.linear(h, raw.tok_emb.weight)
                    else:
                        logits_proj = raw.lm_head(h)
                    logits = raw.logit_softcap * torch.tanh(logits_proj / raw.logit_softcap)
                    batch_loss = F.cross_entropy(logits.float(), y.reshape(-1), reduction="mean")
                else:
                    batch_loss = model(x, y)

            n_tok = y.numel()
            loss_sum += batch_loss.item() * n_tok
            tok_count += n_tok
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = base_bytes[tgt].to(torch.int16)
            tb += (has_space[tgt] & ~is_boundary[prev]).to(torch.int16)
            byte_count += tb.float().sum().item()

    elapsed_ms = (time.time() - t0) * 1000
    val_loss = loss_sum / tok_count
    bpb = (val_loss / math.log(2.0)) * (tok_count / byte_count)
    return val_loss, bpb, elapsed_ms

# ---------------------------------------------------------------------------
# Corrector training
# ---------------------------------------------------------------------------
def train_corrector(
    teacher_model: nn.Module,
    student_model: nn.Module,  # frozen INT4 backbone
    corrector: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    args: Hyperparameters,
    n_steps: int = 200,
    lr: float = 1e-3,
    beta_kd: float = 0.5,
    temperature: float = 2.0,
) -> list[float]:
    """Train corrector on frozen student backbone with teacher distillation."""
    seq_len = args.train_seq_len
    total = val_tokens.numel() - 1
    usable = (total // seq_len) * seq_len

    # Unwrap models
    teacher_raw = teacher_model
    if hasattr(teacher_raw, "_orig_mod"):
        teacher_raw = teacher_raw._orig_mod
    student_raw = student_model
    if hasattr(student_raw, "_orig_mod"):
        student_raw = student_raw._orig_mod

    optimizer = torch.optim.Adam(corrector.parameters(), lr=lr)
    corrector.train()
    teacher_model.eval()
    student_model.eval()

    losses = []
    for step in range(n_steps):
        # Random batch of sequences from val set (we're training the corrector, not the LM)
        max_start = usable - seq_len
        starts = torch.randint(0, max_start, (4,))  # batch of 4
        x_list = [val_tokens[s:s + seq_len] for s in starts]
        y_list = [val_tokens[s + 1:s + seq_len + 1] for s in starts]
        x = torch.stack(x_list).to(device=device, dtype=torch.int64)
        y = torch.stack(y_list).to(device=device, dtype=torch.int64)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # Teacher forward (detached)
            with torch.no_grad():
                t_emb = teacher_raw.tok_emb(x)
                t_emb = F.rms_norm(t_emb, (t_emb.size(-1),))
                t0 = t_emb
                th = t_emb
                t_skips = []
                for i in range(teacher_raw.num_encoder_layers):
                    th = teacher_raw.blocks[i](th, t0)
                    t_skips.append(th)
                for i in range(teacher_raw.num_decoder_layers):
                    if t_skips:
                        th = th + teacher_raw.skip_weights[i].to(dtype=th.dtype)[None, None, :] * t_skips.pop()
                    th = teacher_raw.blocks[teacher_raw.num_encoder_layers + i](th, t0)
                th = teacher_raw.final_norm(th)
                th_flat = th.reshape(-1, th.size(-1))
                if teacher_raw.tie_embeddings:
                    t_logits = F.linear(th_flat, teacher_raw.tok_emb.weight)
                else:
                    t_logits = teacher_raw.lm_head(th_flat)
                t_logits = teacher_raw.logit_softcap * torch.tanh(t_logits / teacher_raw.logit_softcap)

            # Student forward (frozen) + corrector
            with torch.no_grad():
                s_emb = student_raw.tok_emb(x)
                s_emb = F.rms_norm(s_emb, (s_emb.size(-1),))
                s0 = s_emb
                sh = s_emb
                s_skips = []
                for i in range(student_raw.num_encoder_layers):
                    sh = student_raw.blocks[i](sh, s0)
                    s_skips.append(sh)
                for i in range(student_raw.num_decoder_layers):
                    if s_skips:
                        sh = sh + student_raw.skip_weights[i].to(dtype=sh.dtype)[None, None, :] * s_skips.pop()
                    sh = student_raw.blocks[student_raw.num_encoder_layers + i](sh, s0)
                sh = student_raw.final_norm(sh)

            # Corrector (has gradients)
            sh_corr = corrector(sh)
            sh_flat = sh_corr.reshape(-1, sh_corr.size(-1))
            if student_raw.tie_embeddings:
                s_logits = F.linear(sh_flat, student_raw.tok_emb.weight.detach())
            else:
                s_logits = F.linear(sh_flat, student_raw.lm_head.weight.detach())
            s_logits = student_raw.logit_softcap * torch.tanh(s_logits / student_raw.logit_softcap)

        # CE loss
        targets = y.reshape(-1)
        loss_ce = F.cross_entropy(s_logits.float(), targets)

        # KD loss
        loss_kd = F.kl_div(
            F.log_softmax(s_logits.float() / temperature, dim=-1),
            F.softmax(t_logits.float().detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        loss = loss_ce + beta_kd * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  corrector step {step:>4}/{n_steps}  ce:{loss_ce.item():.4f}  kd:{loss_kd.item():.4f}  total:{loss.item():.4f}")
        losses.append(loss.item())

    return losses

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--corrector-steps", type=int, default=200)
    parser.add_argument("--corrector-lr", type=float, default=1e-3)
    cli_args = parser.parse_args()

    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CORRECTOR EXPERIMENT: Is INT4 error cheaply repairable?")
    print("=" * 70)

    # Load tokenizer + val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, args.vocab_size, device)
    print(f"Val tokens: {val_tokens.numel()-1:,}")

    # Load trained model
    print(f"\nLoading checkpoint: {cli_args.checkpoint}")
    teacher_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for m in teacher_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    teacher_model.load_state_dict(torch.load(cli_args.checkpoint, map_location=device))
    teacher_model.eval()

    # =====================================================================
    # STEP 1: Four baselines
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 1: QUANTIZATION BASELINES")
    print("=" * 70)

    results = {}

    # 1a. Float baseline
    print("\n--- Float (no quantization) ---")
    vl, bpb, ms = evaluate_bpb(teacher_model, val_tokens, args.train_seq_len, base_bytes, has_space, is_boundary, device)
    results["float"] = {"bpb": bpb, "bytes": 0, "loss": vl}
    print(f"  val_loss: {vl:.6f}  val_bpb: {bpb:.6f}  time: {ms:.0f}ms")

    # 1b-d. INT8, INT6, INT4
    for bits in [8, 6, 4]:
        print(f"\n--- INT{bits} post-training quantization ---")
        sd = {k: v.detach().cpu() for k, v in teacher_model.state_dict().items()}
        obj, compressed, payload = quantize_sd(sd, bits)
        dq_sd = dequantize_sd(obj, bits)

        q_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        ).to(device).bfloat16()
        for m in q_model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        q_model.load_state_dict(dq_sd, strict=False)
        q_model.eval()

        vl, bpb, ms = evaluate_bpb(q_model, val_tokens, args.train_seq_len, base_bytes, has_space, is_boundary, device)
        artifact_bytes = len(compressed) + 53000  # ~code size
        results[f"int{bits}"] = {"bpb": bpb, "bytes": artifact_bytes, "loss": vl, "compressed": len(compressed)}
        print(f"  val_loss: {vl:.6f}  val_bpb: {bpb:.6f}  time: {ms:.0f}ms")
        print(f"  compressed: {len(compressed):,} bytes  artifact: {artifact_bytes:,} bytes")
        gap = bpb - results["float"]["bpb"]
        print(f"  BPB gap vs float: {gap:+.6f}")

        if bits == 4:
            int4_model = q_model  # keep for corrector training

    # =====================================================================
    # STEP 2: CORRECTOR SWEEP
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 2: CORRECTOR SWEEP ON FROZEN INT4 BACKBONE")
    print("=" * 70)

    int4_bpb = results["int4"]["bpb"]
    float_bpb = results["float"]["bpb"]
    int4_gap = int4_bpb - float_bpb

    print(f"\nINT4 gap to recover: {int4_gap:.6f} BPB")
    print(f"Float: {float_bpb:.6f}  INT4: {int4_bpb:.6f}\n")

    corrector_results = {}
    D = args.model_dim

    for label, corrector_fn, n_params in [
        ("affine", lambda: AffineCorrector(D), 2 * D),
        ("rank-4", lambda: LowRankCorrector(D, 4), 2 * D * 4),
        ("rank-8", lambda: LowRankCorrector(D, 8), 2 * D * 8),
        ("rank-16", lambda: LowRankCorrector(D, 16), 2 * D * 16),
        ("rank-32", lambda: LowRankCorrector(D, 32), 2 * D * 32),
    ]:
        print(f"\n--- {label} ({n_params:,} params, {n_params*2:,} bytes FP16) ---")
        corrector = corrector_fn().to(device).float()

        # Train corrector with teacher distillation
        train_corrector(
            teacher_model, int4_model, corrector, val_tokens, device, args,
            n_steps=cli_args.corrector_steps, lr=cli_args.corrector_lr,
        )

        # Evaluate
        vl, bpb, ms = evaluate_bpb(
            int4_model, val_tokens, args.train_seq_len,
            base_bytes, has_space, is_boundary, device,
            corrector=corrector,
        )
        recovery = int4_bpb - bpb
        helper_bytes = n_params * 2  # FP16
        corrector_results[label] = {"bpb": bpb, "recovery": recovery, "helper_bytes": helper_bytes}
        print(f"  val_bpb: {bpb:.6f}  recovery: {recovery:+.6f}  ({recovery/int4_gap*100:.1f}% of gap)")
        print(f"  helper size: {helper_bytes:,} bytes")

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Bytes':>12} {'BPB':>10} {'vs INT4':>10} {'vs Float':>10}")
    print("-" * 62)
    for name, r in results.items():
        gap_float = r["bpb"] - results["float"]["bpb"]
        print(f"{name:<20} {r.get('compressed', 0):>12,} {r['bpb']:>10.6f} {'':>10} {gap_float:>+10.6f}")
    for name, r in corrector_results.items():
        int4_bytes = results["int4"]["compressed"]
        total = int4_bytes + r["helper_bytes"]
        gap_float = r["bpb"] - results["float"]["bpb"]
        print(f"INT4+{name:<14} {total:>12,} {r['bpb']:>10.6f} {r['recovery']:>+10.6f} {gap_float:>+10.6f}")

    print(f"\nKey question: Does INT4 + corrector beat INT6 at fewer total bytes?")
    if "int6" in results and corrector_results:
        best_corrector = min(corrector_results.values(), key=lambda x: x["bpb"])
        int6_bpb = results["int6"]["bpb"]
        best_bpb = best_corrector["bpb"]
        print(f"  INT6: {int6_bpb:.6f} BPB at {results['int6']['compressed']:,} bytes")
        print(f"  Best corrector: {best_bpb:.6f} BPB at {results['int4']['compressed'] + best_corrector['helper_bytes']:,} bytes")
        if best_bpb < int6_bpb:
            print(f"  => YES, corrector wins by {int6_bpb - best_bpb:.6f} BPB")
        else:
            print(f"  => NO, INT6 is better by {best_bpb - int6_bpb:.6f} BPB")


if __name__ == "__main__":
    main()
