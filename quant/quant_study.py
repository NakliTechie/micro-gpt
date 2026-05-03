"""
Quantization study using the TALOS-V2 trained microGPT weights.

For each quantization format we:
  1. Round the fp32 weights to the nearest representable value, dequantize back
     to fp32 for inference (so we measure *only* the precision loss).
  2. Score the model on the first 500 names (average NLL per character).
  3. Generate 5 samples for visual inspection.

We compare:
  - fp32 (baseline)
  - Q4.12 (TALOS, strict scale=4096, int16) -- this is what the FPGA uses
  - Q3.13 (16-bit, half the integer range)
  - per-tensor symmetric int8 (~Q-equivalent: ~7 fractional bits)
  - per-tensor symmetric int4 (3 levels of magnitude)
  - per-tensor symmetric int2 (sign + 1 magnitude bit)

Per-tensor scaling: scale each weight matrix independently by max|w|/levels.
This is what practical fixed-point inference does -- it gets way more out of
a small int range than a hard "Q4.12 for all" because most of the weights
are in [-0.5, 0.5] and Q4.12 wastes bits on the [-8,-0.5] U [0.5,8] range
nothing ever uses.
"""
import sys
import time
from pathlib import Path

import numpy as np

ASSETS = Path("benchmark/talos-vs-macbook-m5-pro/assets")
WEIGHTS_PATH = ASSETS / "weights_only.npy"
NAMES_PATH = ASSETS / "names.txt"

VOCAB_SIZE = 27
BLOCK_SIZE = 16
N_HEAD = 4
N_EMBD = 16
HEAD_DIM = N_EMBD // N_HEAD
BOS = 26
INV_SQRT_HD = np.float32(1.0 / np.sqrt(HEAD_DIM))

WEIGHT_KEYS = (
    "wte", "wpe",
    "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo",
    "layer0.mlp_fc1", "layer0.mlp_fc2",
    "lm_head",
)


def load_weights():
    raw = np.load(WEIGHTS_PATH, allow_pickle=True).item()
    return {k: np.asarray(raw[k], dtype=np.float32) for k in WEIGHT_KEYS}


# ---------- forward pass (matches bench_numpy.py exactly) ----------

def rmsnorm(x):
    return x * np.float32(1.0) / np.sqrt((x * x).mean() + np.float32(1e-5))


def forward(W, tok, pos, K, V):
    """Single-step forward. Updates K[pos] and V[pos] in place. Returns logits."""
    x = W["wte"][tok] + W["wpe"][pos]
    x = rmsnorm(x)

    xr = x
    x = rmsnorm(x)
    q = W["layer0.attn_wq"] @ x
    k = W["layer0.attn_wk"] @ x
    v = W["layer0.attn_wv"] @ x
    K[pos] = k
    V[pos] = v

    Kt = K[: pos + 1].reshape(pos + 1, N_HEAD, HEAD_DIM)
    Vt = V[: pos + 1].reshape(pos + 1, N_HEAD, HEAD_DIM)
    qh = q.reshape(N_HEAD, HEAD_DIM)
    logits_attn = np.einsum("hd,thd->ht", qh, Kt) * INV_SQRT_HD
    logits_attn -= logits_attn.max(axis=1, keepdims=True)
    np.exp(logits_attn, out=logits_attn)
    logits_attn /= logits_attn.sum(axis=1, keepdims=True)
    head_out = np.einsum("ht,thd->hd", logits_attn, Vt).reshape(N_EMBD)

    x = W["layer0.attn_wo"] @ head_out
    x = x + xr

    xr = x
    x = rmsnorm(x)
    h = W["layer0.mlp_fc1"] @ x
    np.maximum(h, 0, out=h)
    x = W["layer0.mlp_fc2"] @ h
    x = x + xr

    return W["lm_head"] @ x


# ---------- quantization functions ----------

def q412_strict(W):
    """Exactly what TALOS does: scale=4096, int16 range, no per-tensor scaling."""
    out = {}
    for k, w in W.items():
        q = np.clip(np.round(w * 4096.0), -32768, 32767).astype(np.int16)
        out[k] = (q.astype(np.float32) / 4096.0)
    return out


def q3_13_strict(W):
    """16-bit with one less integer bit -- max range +/- 4."""
    out = {}
    for k, w in W.items():
        q = np.clip(np.round(w * 8192.0), -32768, 32767).astype(np.int16)
        out[k] = (q.astype(np.float32) / 8192.0)
    return out


def per_tensor_sym(W, n_bits):
    """Symmetric per-tensor quantization. For each tensor, scale = max|w|/levels."""
    levels = 2 ** (n_bits - 1) - 1   # e.g. 127 for int8, 7 for int4, 1 for int2
    out = {}
    for k, w in W.items():
        m = np.abs(w).max()
        if m == 0 or levels == 0:
            out[k] = w.copy()
            continue
        scale = m / levels
        q = np.clip(np.round(w / scale), -levels - 1, levels)
        out[k] = (q * scale).astype(np.float32)
    return out


# ---------- scoring ----------

def score_nll(W, names, max_names=500):
    """Average NLL per character on the first max_names names."""
    K = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
    V = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
    total_nll = 0.0
    total_n = 0
    for name in names[:max_names]:
        tokens = [BOS] + [ord(c) - ord("a") for c in name] + [BOS]
        pos = 0
        for i in range(len(tokens) - 1):
            if pos >= BLOCK_SIZE:
                break
            tok = tokens[i]
            target = tokens[i + 1]
            logits = forward(W, tok, pos, K, V)
            z = logits - logits.max()
            log_norm = np.log(np.exp(z).sum())
            log_p_target = z[target] - log_norm
            total_nll += -log_p_target
            total_n += 1
            pos += 1
        # reset for next name
        K.fill(0); V.fill(0)
    return float(total_nll / total_n)


def sample_names(W, n=5, seed=42):
    """Generate n names from the (quantized) model."""
    import random
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(n):
        K = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
        V = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
        tok = BOS
        s = []
        for pos in range(BLOCK_SIZE):
            logits = forward(W, tok, pos, K, V).copy() * np.float32(1.0 / 0.5)  # temp=0.5
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            tok = rng.choices(range(VOCAB_SIZE), weights=probs.tolist())[0]
            if tok == BOS:
                break
            s.append(chars[tok])
        out.append("".join(s))
    return out


def reconstruction_error(W_ref, W_q):
    errs = [np.abs(W_ref[k] - W_q[k]).max() for k in W_ref]
    return max(errs)


# ---------- main ----------

def main():
    W_fp32 = load_weights()
    names = NAMES_PATH.read_text().splitlines()
    print(f"loaded {len(names):,} names; using first 500 for scoring")
    print(f"baseline params: {sum(w.size for w in W_fp32.values()):,}")
    print()

    cases = [
        ("fp32 (baseline)",      W_fp32),
        ("Q4.12 strict (TALOS)", q412_strict(W_fp32)),
        ("Q3.13 strict",         q3_13_strict(W_fp32)),
        ("per-tensor int8 sym",  per_tensor_sym(W_fp32, 8)),
        ("per-tensor int4 sym",  per_tensor_sym(W_fp32, 4)),
        ("per-tensor int2 sym",  per_tensor_sym(W_fp32, 2)),
    ]

    print(f"{'format':24s}  {'max|err|':>10s}  {'NLL':>6s}  {'time':>6s}  samples")
    print("-" * 90)
    for label, W in cases:
        err = reconstruction_error(W_fp32, W)
        t0 = time.time()
        nll = score_nll(W, names)
        elapsed = time.time() - t0
        samps = sample_names(W, n=5)
        print(f"{label:24s}  {err:10.5f}  {nll:6.4f}  {elapsed:5.1f}s  {samps}")


if __name__ == "__main__":
    main()
