"""
Step 6: GPT with multi-head self-attention + Adam.

Multi-head: split D=16 into H=4 heads of head_dim=4. Each head learns its
own attention pattern in parallel; the results are concatenated and projected.
The same model capacity, but more flexible attention.

  q, k, v: (B, T, D)  -- as before
  reshape: (B, T, H, head_dim)
  transpose: (B, H, T, head_dim)
  scores = q @ k^T / sqrt(head_dim)  shape (B, H, T, T)
  mask + softmax + (attn @ v) -> (B, H, T, head_dim)
  transpose + reshape back to (B, T, D), then W_o.

Adam: per-parameter exponential moving averages of gradient (m) and squared
gradient (v). The update divides by sqrt(v), which gives each parameter its
own effective learning rate. Crucial for transformers -- vanilla SGD plateaus
around 2.30 on this dataset; Adam pushes through to ~2.00.

Plus: LR warmup over the first 200 steps, then constant.

Note: this is an *educational variant* of the microGPT architecture; it has
learnable RMSNorm gains and a final RMSNorm before the LM head, giving it
4,240 params. The TALOS-V2 reference model used in the WASM benchmark has
parameter-free RMSNorm and no final norm before the LM head, giving 4,192
params. Same family, slightly different parameter count.
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
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD          # = 4
MLP_HIDDEN = 4 * N_EMBD              # = 64


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
    B, T = idx.shape
    tok = p["token_emb"].embedding(idx)              # (B, T, D)
    pos = p["pos_emb"].embedding(np.arange(T))       # (T, D)
    x = tok + pos                                    # (B, T, D)

    # --- multi-head attention ---
    nx = x.rms_norm(p["rms1_g"])
    q = nx @ p["W_q"]
    k = nx @ p["W_k"]
    v = nx @ p["W_v"]
    # split into heads: (B, T, D) -> (B, T, H, head_dim) -> (B, H, T, head_dim)
    q = q.reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)

    kT = k.transpose(0, 1, 3, 2)                     # (B, H, head_dim, T)
    scores = (q @ kT) * (1.0 / np.sqrt(HEAD_DIM))    # (B, H, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = scores.masked_fill(mask, -1e9)
    attn = scores.softmax(axis=-1)
    out = attn @ v                                   # (B, H, T, head_dim)
    # back to (B, T, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, N_EMBD)
    out = out @ p["W_o"]
    x = x + out                                      # residual

    # --- mlp ---
    nx = x.rms_norm(p["rms2_g"])
    h = (nx @ p["W_mlp1"]).relu()
    m = h @ p["W_mlp2"]
    x = x + m

    # --- output ---
    x = x.rms_norm(p["rmsf_g"])
    return x @ p["lm_head"]                          # (B, T, V)


class Adam:
    """Per-parameter EMA of gradient (m) and squared gradient (v)."""

    def __init__(self, params: dict, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}

    def step(self, lr_override=None):
        self.t += 1
        lr = lr_override if lr_override is not None else self.lr
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, p in self.params.items():
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            p.data -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


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


def get_lr(step, max_lr=3e-3, warmup=200):
    return max_lr * min(1.0, (step + 1) / warmup)


def main():
    rng = np.random.default_rng(42)
    names = Path("data/names.txt").read_text().splitlines()
    rng.shuffle(names)

    # Real train/val split: 90/10. The stream is built per-split so the
    # validation tokens are drawn only from names the model never trained on.
    split = int(0.9 * len(names))
    train_names, val_names = names[:split], names[split:]
    train_stream = build_stream(train_names)
    val_stream = build_stream(val_names)
    print(f"train stream: {len(train_stream):,} tokens ({len(train_names):,} names)")
    print(f"val   stream: {len(val_stream):,} tokens ({len(val_names):,} names)")

    p = init_params(rng)
    n_params = sum(t.data.size for t in p.values())
    print(f"parameters: {n_params:,}  (block_size={BLOCK_SIZE}, n_embd={N_EMBD}, n_head={N_HEAD})")

    optim = Adam(p, lr=3e-3)
    batch_size = 32
    steps = 5000

    t0 = time.time()
    for step in range(steps):
        xb, yb = get_batch(train_stream, batch_size, BLOCK_SIZE, rng)

        for t in p.values():
            t.grad = np.zeros_like(t.data)

        logits = forward(p, xb)
        loss = logits.cross_entropy(yb)
        loss.backward()

        optim.step(lr_override=get_lr(step))

        if step % 250 == 0 or step == steps - 1:
            print(f"step {step:5d} | lr {get_lr(step):.4f} | loss {float(loss.data):.4f}")

    print(f"\ntraining took {time.time() - t0:.1f}s")
    train_nll = estimate_loss(p, train_stream)
    val_nll = estimate_loss(p, val_stream)
    print(f"train NLL: {train_nll:.4f}")
    print(f"val   NLL: {val_nll:.4f}    <- truly held-out (90/10 name-level split)")
    print(f"step 5 single-head SGD: 2.29")
    print(f"step 4 MLP baseline:    2.20")

    # Save trained weights for the quantization phase
    np.savez(
        "ladder/step6_weights.npz",
        **{k: t.data for k, t in p.items()},
    )
    print(f"\nweights saved to ladder/step6_weights.npz")

    py_rng = random.Random(42)
    print(f"\n--- 20 samples ---")
    for i in range(20):
        print(f"sample {i + 1:2d}: {sample(p, py_rng)}")


if __name__ == "__main__":
    main()
