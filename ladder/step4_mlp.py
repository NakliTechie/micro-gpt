"""
Step 4: MLP Language Model.

Three things change vs Step 3:

  1. Embeddings: (V, D) learnable table replaces one-hot input. Similar
     characters can now share gradient signal.
  2. Context window: input is the previous BLOCK_SIZE chars, not just one.
  3. Hidden layer + tanh: real nonlinearity, more than a glorified bigram.

Forward:
    inputs (B, 3) int  -> emb (B, 3, 10) -> flat (B, 30)
                       -> tanh(flat @ W1 + b1)  (B, 64)
                       -> hidden @ W2 + b2      (B, 27) logits
                       -> cross_entropy(logits, targets)

This is structurally Karpathy's makemore MLP (Bengio et al, 2003).
Should beat the bigram (~2.45) by a clear margin -- expect ~2.0-2.1.
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
BLOCK_SIZE = 3
EMBED_DIM = 10
HIDDEN_DIM = 64


def build_dataset(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for name in names:
        ids = [ord(c) - ord("a") for c in name] + [BOS]
        padded = [BOS] * BLOCK_SIZE + ids
        for i in range(len(ids)):
            xs.append(padded[i:i + BLOCK_SIZE])
            ys.append(padded[i + BLOCK_SIZE])
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def full_dataset_nll(xs, ys, embed, W1, b1, W2, b2, batch=1024):
    total_nll = 0.0
    total_n = 0
    for i in range(0, len(xs), batch):
        xb, yb = xs[i:i + batch], ys[i:i + batch]
        e = embed.data[xb]
        f = e.reshape(len(xb), -1)
        h = np.tanh(f @ W1.data + b1.data)
        l = h @ W2.data + b2.data
        z = l - l.max(axis=-1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        total_nll += -lp[np.arange(len(yb)), yb].sum()
        total_n += len(yb)
    return total_nll / total_n


def main():
    rng = np.random.default_rng(42)
    names = Path("data/names.txt").read_text().splitlines()
    xs, ys = build_dataset(names)
    N = len(xs)
    print(f"dataset: {N:,} (context, target) pairs from {len(names):,} names")

    embed = Tensor(rng.standard_normal((VOCAB_SIZE, EMBED_DIM)) * 0.1)
    fan_in = BLOCK_SIZE * EMBED_DIM
    W1 = Tensor(rng.standard_normal((fan_in, HIDDEN_DIM)) * (5.0 / 3.0) / np.sqrt(fan_in))
    b1 = Tensor(np.zeros(HIDDEN_DIM))
    W2 = Tensor(rng.standard_normal((HIDDEN_DIM, VOCAB_SIZE)) * 0.01)
    b2 = Tensor(np.zeros(VOCAB_SIZE))
    params = [embed, W1, b1, W2, b2]
    print(f"parameters: {sum(p.data.size for p in params):,}")

    batch_size = 64
    steps = 30000
    base_lr = 0.1

    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, N, size=batch_size)
        xb, yb = xs[idx], ys[idx]

        for p in params:
            p.grad = np.zeros_like(p.data)

        emb = embed.embedding(xb)
        flat = emb.reshape(batch_size, fan_in)
        h = (flat @ W1 + b1).tanh()
        logits = h @ W2 + b2
        loss = logits.cross_entropy(yb)
        loss.backward()

        lr = base_lr if step < 20000 else base_lr * 0.1
        for p in params:
            p.data -= lr * p.grad

        if step % 2500 == 0 or step == steps - 1:
            print(f"step {step:6d} | mini-batch loss {float(loss.data):.4f}")

    print(f"\ntraining took {time.time() - t0:.1f}s")
    full_nll = full_dataset_nll(xs, ys, embed, W1, b1, W2, b2)
    print(f"full-dataset NLL: {full_nll:.4f}")
    print(f"step 1 bigram baseline: 2.4546")

    py_rng = random.Random(42)
    print(f"\n--- 10 samples ---")
    for i in range(10):
        out = []
        ctx = [BOS] * BLOCK_SIZE
        for _ in range(20):
            xb = np.array([ctx])
            e = embed.data[xb].reshape(1, fan_in)
            h = np.tanh(e @ W1.data + b1.data)
            l = (h @ W2.data + b2.data)[0]
            z = l - l.max()
            p = np.exp(z)
            p /= p.sum()
            r = py_rng.random()
            cum = 0.0
            for j, pj in enumerate(p):
                cum += pj
                if r < cum:
                    nxt = j
                    break
            if nxt == BOS:
                break
            out.append(chr(nxt + ord("a")))
            ctx = ctx[1:] + [nxt]
        print(f"sample {i + 1:2d}: {''.join(out)}")


if __name__ == "__main__":
    main()
