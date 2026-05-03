"""
Step 2: Neural Bigram with Manual Gradients.

Learn the same bigram probabilities via gradient descent.

Architecture:
    logits[i, k] = W[k, x[i]]             # one column per input char
    probs[i, :]  = softmax(logits[i, :])
    loss         = -log(probs[i, y[i]])    # cross-entropy

Manual gradients (the key identity):
    dL/dlogits = probs - one_hot(y)
    dL/dW[:, x_i] = dL/dlogits[i]          # only the input column gets gradient

Should converge to roughly the same 2.45 NLL as Step 1's counts.
"""
import random
from pathlib import Path

import numpy as np

VOCAB_SIZE = 27
BOS = 26


def encode(name: str) -> list[int]:
    return [BOS] + [ord(c) - ord("a") for c in name] + [BOS]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def build_dataset(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for name in names:
        ids = encode(name)
        for prev, nxt in zip(ids, ids[1:]):
            xs.append(prev)
            ys.append(nxt)
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def main():
    rng = np.random.default_rng(42)
    names = Path("data/names.txt").read_text().splitlines()
    xs, ys = build_dataset(names)
    N = len(xs)
    print(f"dataset: {N:,} (input, target) pairs")

    W = rng.standard_normal((VOCAB_SIZE, VOCAB_SIZE)) * 0.01
    lr = 50.0
    epochs = 200

    for epoch in range(epochs):
        logits = W[:, xs].T                    # (N, 27)
        probs = softmax(logits)                # (N, 27)
        loss = -np.log(probs[np.arange(N), ys] + 1e-12).mean()

        dlogits = probs.copy()
        dlogits[np.arange(N), ys] -= 1.0
        dlogits /= N                           # because loss is the mean

        dW = np.zeros_like(W)
        np.add.at(dW, (slice(None), xs), dlogits.T)

        W -= lr * dW

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} | loss {loss:.4f}")

    print(f"\nfinal loss: {loss:.4f}")
    print(f"step 1 counting baseline: 2.4546")

    py_rng = random.Random(42)
    print(f"\n--- 10 samples ---")
    for i in range(10):
        out = []
        cur = BOS
        for _ in range(20):
            p = softmax(W[:, cur])
            r = py_rng.random()
            cum = 0.0
            for j, pj in enumerate(p):
                cum += pj
                if r < cum:
                    cur = j
                    break
            if cur == BOS:
                break
            out.append("." if cur == BOS else chr(cur + ord("a")))
        print(f"sample {i+1:2d}: {''.join(out)}")


if __name__ == "__main__":
    main()
