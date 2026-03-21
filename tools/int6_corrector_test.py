#!/usr/bin/env python3
"""
Quick test: INT6 + corrector. Does a tiny corrector close the 0.024 BPB INT6 gap?

Usage: python3 tools/int6_corrector_test.py --checkpoint final_model.pt
"""
from __future__ import annotations
import argparse, io, math, sys, time, zlib
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, RMSNorm,
    load_validation_tokens, build_sentencepiece_luts,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    INT8_KEEP_FLOAT_MAX_NUMEL, INT8_KEEP_FLOAT_STORE_DTYPE, INT8_CLIP_Q,
)

# INT6 quantize/dequantize
def quantize_int6(sd):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    for name, t in sd.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            pt = t.to(INT8_KEEP_FLOAT_STORE_DTYPE) if t.is_floating_point() and t.dtype in {torch.float32, torch.bfloat16} else t
            passthrough[name] = pt
            continue
        t32 = t.float()
        if t32.ndim == 2:
            clip = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            s = (clip / 31.0).clamp_min(1.0 / 31.0)
            q = torch.clamp(torch.round(t32 / s[:, None]), -32, 31).to(torch.int8)
        else:
            clip = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
            s = torch.tensor(clip / 31.0 if clip > 0 else 1.0)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / s), -32, 31).to(torch.int8)
        quantized[name] = q.contiguous()
        scales[name] = s.to(torch.float16).contiguous() if s.ndim > 0 else s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}

def dequantize_int6(obj):
    out = {}
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if s.ndim > 0:
            out[name] = (q.float() * s.float()[:, None]).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out[name] = t.detach().cpu().contiguous()
    return out

# Correctors
class AffineCorrector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, h):
        return self.a * h + self.b

class LowRankCorrector(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self, h):
        return h + self.up(self.down(h))

# Eval
def eval_bpb(raw, val_tokens, seq_len, bb, hs, ib, device, corrector=None):
    total = val_tokens.numel() - 1
    usable = (total // seq_len) * seq_len
    lsum, tc, bc = 0.0, 0, 0.0
    with torch.inference_mode():
        for s in range(0, usable, seq_len * 8):
            e = min(s + seq_len * 8, usable)
            n = (e - s) // seq_len
            local = val_tokens[s:s + n * seq_len + 1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(n, seq_len), local[1:].reshape(n, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if corrector is None:
                    loss = raw(x, y)
                else:
                    # Manual forward with corrector after final_norm
                    xe = F.rms_norm(raw.tok_emb(x), (raw.tok_emb.weight.shape[1],))
                    x0, h, skips = xe, xe, []
                    for i in range(raw.num_encoder_layers):
                        h = raw.blocks[i](h, x0); skips.append(h)
                    for i in range(raw.num_decoder_layers):
                        if skips:
                            h = h + raw.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
                        h = raw.blocks[raw.num_encoder_layers + i](h, x0)
                    h = raw.final_norm(h)
                    h = corrector(h)
                    hf = h.reshape(-1, h.size(-1))
                    lp = F.linear(hf, raw.tok_emb.weight) if raw.tie_embeddings else raw.lm_head(hf)
                    logits = raw.logit_softcap * torch.tanh(lp / raw.logit_softcap)
                    loss = F.cross_entropy(logits.float(), y.reshape(-1))
            lsum += loss.item() * y.numel(); tc += y.numel()
            tb = bb[y.reshape(-1)].to(torch.int16)
            tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
            bc += tb.float().sum().item()
    vl = lsum / tc
    return vl, (vl / math.log(2.0)) * (tc / bc)

# Train corrector (CE-only on frozen INT6 backbone)
def train_corr(raw, corrector, val_tokens, seq_len, device, steps=300, lr=1e-3):
    usable = ((val_tokens.numel() - 1) // seq_len) * seq_len
    opt = torch.optim.Adam(corrector.parameters(), lr=lr)
    corrector.train()
    # Clear RoPE caches
    for block in raw.blocks:
        block.attn.rotary._cos_cached = None
        block.attn.rotary._sin_cached = None
        block.attn.rotary._seq_len_cached = 0
    for step in range(steps):
        starts = torch.randint(0, usable - seq_len, (4,))
        x = torch.stack([val_tokens[s:s+seq_len] for s in starts]).to(device=device, dtype=torch.int64)
        y = torch.stack([val_tokens[s+1:s+seq_len+1] for s in starts]).to(device=device, dtype=torch.int64)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            with torch.no_grad():
                xe = F.rms_norm(raw.tok_emb(x), (raw.tok_emb.weight.shape[1],))
                x0, h, skips = xe, xe, []
                for i in range(raw.num_encoder_layers):
                    h = raw.blocks[i](h, x0); skips.append(h)
                for i in range(raw.num_decoder_layers):
                    if skips:
                        h = h + raw.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
                    h = raw.blocks[raw.num_encoder_layers + i](h, x0)
                h = raw.final_norm(h)
            # Corrector with gradients
            h = corrector(h.detach())
            hf = h.reshape(-1, h.size(-1))
            with torch.no_grad():
                w = raw.tok_emb.weight if raw.tie_embeddings else raw.lm_head.weight
            lp = F.linear(hf, w.detach())
            logits = raw.logit_softcap * torch.tanh(lp / raw.logit_softcap)
            loss = F.cross_entropy(logits.float(), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"    step {step:>4}/{steps}  loss: {loss.item():.4f}")

def build_model(args, device):
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for mod in m.modules():
        if isinstance(mod, CastedLinear): mod.float()
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final_model.pt")
    parser.add_argument("--steps", type=int, default=300)
    cli = parser.parse_args()
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = args.model_dim

    print("=" * 60)
    print("INT6 + CORRECTOR TEST")
    print("=" * 60)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb, hs, ib = build_sentencepiece_luts(sp, args.vocab_size, device)

    # Load and quantize to INT6
    teacher = build_model(args, device)
    teacher.load_state_dict(torch.load(cli.checkpoint, map_location=device))
    teacher.eval()

    vl, bpb = eval_bpb(teacher, val_tokens, args.train_seq_len, bb, hs, ib, device)
    print(f"\n  float:  bpb={bpb:.6f}")

    sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
    obj = quantize_int6(sd)
    dq = dequantize_int6(obj)
    int6_model = build_model(args, device)
    int6_model.load_state_dict(dq, strict=False)
    int6_model.eval()

    vl, int6_bpb = eval_bpb(int6_model, val_tokens, args.train_seq_len, bb, hs, ib, device)
    int6_gap = int6_bpb - bpb
    print(f"  INT6:   bpb={int6_bpb:.6f}  gap={int6_gap:+.6f}")
    print(f"\n  INT6 gap to recover: {int6_gap:.6f} BPB")

    # Corrector sweep
    print(f"\n{'='*60}")
    print("CORRECTOR SWEEP ON INT6")
    print(f"{'='*60}")

    for label, make_corr, n_params in [
        ("affine", lambda: AffineCorrector(D), 2 * D),
        ("rank-4", lambda: LowRankCorrector(D, 4), 2 * D * 4),
        ("rank-8", lambda: LowRankCorrector(D, 8), 2 * D * 8),
        ("rank-16", lambda: LowRankCorrector(D, 16), 2 * D * 16),
        ("rank-32", lambda: LowRankCorrector(D, 32), 2 * D * 32),
    ]:
        helper_bytes = n_params * 2
        print(f"\n  --- {label} ({n_params:,} params, {helper_bytes:,} bytes) ---")
        corr = make_corr().to(device).float()
        train_corr(int6_model, corr, val_tokens, args.train_seq_len, device, steps=cli.steps)
        corr.eval()
        vl, cbpb = eval_bpb(int6_model, val_tokens, args.train_seq_len, bb, hs, ib, device, corrector=corr)
        recovery = int6_bpb - cbpb
        pct = recovery / int6_gap * 100 if int6_gap > 0 else 0
        print(f"    bpb={cbpb:.6f}  recovery={recovery:+.6f} ({pct:.1f}% of gap)  helper={helper_bytes:,} bytes")

    print(f"\n  float: {bpb:.6f}  INT6: {int6_bpb:.6f}  gap: {int6_gap:.6f}")

if __name__ == "__main__":
    main()
