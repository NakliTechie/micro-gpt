"""
Step 5: GPT with single-head self-attention.

The architecture jumps from "predict next char from a fixed window" to
"process a whole sequence in parallel, every position predicting its
successor."

  inputs (B, T) int -> token_emb + pos_emb -> (B, T, D)
                    -> RMSNorm
                    -> Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
                    -> scores = Q K^T / sqrt(D), causal-masked, softmax
                    -> attn @ V @ W_o
                    -> + residual
                    -> RMSNorm
                    -> ReLU MLP (4x expansion) + residual
                    -> RMSNorm
                    -> lm_head -> logits (B, T, V)

This matches the microGPT / TALOS-V2 architecture (single head version):
block_size=16, n_embd=16, mlp_hidden=64. About 4K parameters.

Training: random windows from a concatenated stream of names separated by
BOS markers. Each window of length T+1 yields T training signals (shift by 1).

Targets a meaningful improvement over the MLP (~2.20).
"""
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tensor import Tensor

VOCAB_SIZE = 27
BOS = 26
BLOCK_SIZE = 16
N_EMBD = 16
N_HEAD = 1
MLP_HIDDEN = 4 * N_EMBD


def init_params(rng):
    s = lambda shape, scale: Tensor(rng.standard_normal(shape) * scale)
    inv_d = 1.0 / np.sqrt(N_EMBD)
    inv_h = 1.0 / np.sqrt(MLP_HIDDEN)
    return {
        "token_emb": s((VOCAB_SIZE, N_EMBD), 0.1),
        "pos_emb":   s((BLOCK_SIZE, N_EMBD), 0.02),
        "rms1_g":    Tensor(np.ones(N_EMBD)),
        "W_q":       s((N_EMBD, N_EMBD), inv_d),
        "W_k":       s((N_EMBD, N_EMBD), inv_d),
        "W_v":       s((N_EMBD, N_EMBD), inv_d),
        "W_o":       s((N_EMBD, N_EMBD), inv_d),
        "rms2_g":    Tensor(np.ones(N_EMBD)),
        "W_mlp1":    s((N_EMBD, MLP_HIDDEN), inv_d),
        "W_mlp2":    s((MLP_HIDDEN, N_EMBD), inv_h),
        "rmsf_g":    Tensor(np.ones(N_EMBD)),
        "lm_head":   s((N_EMBD, VOCAB_SIZE), inv_d),
    }


def forward(p, idx):
    """idx: (B, T) int tokens. Returns logits (B, T, V)."""
    B, T = idx.shape
    tok = p["token_emb"].embedding(idx)              # (B, T, D)
    pos = p["pos_emb"].embedding(np.arange(T))       # (T, D)
    x = tok + pos                                    # (B, T, D); pos broadcasts

    # --- attention sub-layer ---
    nx = x.rms_norm(p["rms1_g"])
    q = nx @ p["W_q"]                                # (B, T, D)
    k = nx @ p["W_k"]
    v = nx @ p["W_v"]
    kT = k.transpose(0, 2, 1)                        # (B, D, T)
    scores = (q @ kT) * (1.0 / np.sqrt(N_EMBD))      # (B, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1) # True above diag
    scores = scores.masked_fill(mask, -1e9)
    attn = scores.softmax(axis=-1)
    a = (attn @ v) @ p["W_o"]                        # (B, T, D)
    x = x + a                                        # residual

    # --- mlp sub-layer ---
    nx = x.rms_norm(p["rms2_g"])
    h = (nx @ p["W_mlp1"]).relu()                    # (B, T, 4D)
    m = h @ p["W_mlp2"]                              # (B, T, D)
    x = x + m                                        # residual

    # --- output ---
    x = x.rms_norm(p["rmsf_g"])
    return x @ p["lm_head"]                          # (B, T, V)


def build_stream(names):
    stream = [BOS]
    for name in names:
        stream.extend(ord(c) - ord("a") for c in name)
        stream.append(BOS)
    return np.array(stream, dtype=np.int64)


def get_batch(stream, batch_size, block_size, rng):
    starts = rng.integers(0, len(stream) - block_size - 1, size=batch_size)
    x = np.stack([stream[s:s + block_size] for s in starts])
    y = np.stack([stream[s + 1:s + block_size + 1] for s in starts])
    return x, y


def estimate_loss(p, stream, batches=20, batch_size=64):
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(batches):
        xb, yb = get_batch(stream, batch_size, BLOCK_SIZE, rng)
        logits = forward(p, xb)
        total += float(logits.cross_entropy(yb).data)
    return total / batches


def sample(p, rng_py):
    ctx = [BOS]
    out = []
    for _ in range(20):
        ctx_tail = ctx[-BLOCK_SIZE:]
        xb = np.array([ctx_tail])
        logits = forward(p, xb).data[0, -1]
        z = logits - logits.max()
        probs = np.exp(z); probs /= probs.sum()
        r = rng_py.random()
        cum = 0.0
        nxt = VOCAB_SIZE - 1
        for j, pj in enumerate(probs):
            cum += pj
            if r < cum:
                nxt = j
                break
        if nxt == BOS:
            if out:
                break
            ctx.append(nxt)
            continue
        out.append(chr(nxt + ord("a")))
        ctx.append(nxt)
    return "".join(out)


def main():
    rng = np.random.default_rng(42)
    names = Path("data/names.txt").read_text().splitlines()
    rng.shuffle(names)
    stream = build_stream(names)
    print(f"stream length: {len(stream):,} tokens")

    p = init_params(rng)
    n_params = sum(t.data.size for t in p.values())
    print(f"parameters: {n_params:,}")

    batch_size = 32
    steps = 5000
    base_lr = 0.05

    t0 = time.time()
    for step in range(steps):
        xb, yb = get_batch(stream, batch_size, BLOCK_SIZE, rng)

        for t in p.values():
            t.grad = np.zeros_like(t.data)

        logits = forward(p, xb)
        loss = logits.cross_entropy(yb)
        loss.backward()

        lr = base_lr if step < int(0.7 * steps) else base_lr * 0.1
        for t in p.values():
            t.data -= lr * t.grad

        if step % 250 == 0 or step == steps - 1:
            print(f"step {step:5d} | loss {float(loss.data):.4f}")

    print(f"\ntraining took {time.time() - t0:.1f}s")
    held_out = estimate_loss(p, stream)
    print(f"held-out NLL: {held_out:.4f}")
    print(f"step 4 MLP baseline: 2.20")

    py_rng = random.Random(42)
    print(f"\n--- 10 samples ---")
    for i in range(10):
        print(f"sample {i + 1:2d}: {sample(p, py_rng)}")


if __name__ == "__main__":
    main()
