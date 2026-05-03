"""
Step 3: Autograd.

Build a tiny tensor library that tracks how each tensor was created and can
walk the graph backward to compute gradients automatically.

Every Tensor has:
  - data: the numpy array
  - grad: accumulated dL/d(self)
  - _children: parents in the graph
  - _backward: function that given out.grad, accumulates child.grad

loss.backward() does a topological sort and calls every _backward in reverse.

Replaces Step 2's hand-derived dL/dW with the same math, computed by walking
the graph. Final loss should match Step 2 (~2.46).
"""
import random
from pathlib import Path

import numpy as np

VOCAB_SIZE = 27
BOS = 26


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._children = _children
        self._backward = lambda: None
        self._op = _op

    def gather_cols(self, idx):
        out = Tensor(self.data[:, idx], (self,), "gather_cols")

        def _backward():
            np.add.at(self.grad, (slice(None), idx), out.grad)

        out._backward = _backward
        return out

    def transpose(self):
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        # self has shape (N, V) -- treat as logits.
        z = self.data - self.data.max(axis=-1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        N = self.data.shape[0]
        loss_val = -log_probs[np.arange(N), targets].mean()
        out = Tensor(loss_val, (self,), "cross_entropy")

        def _backward():
            probs = np.exp(log_probs)
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1.0
            grad /= N
            self.grad += grad * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)  # dL/dL = 1
        for v in reversed(topo):
            v._backward()


def encode(name):
    return [BOS] + [ord(c) - ord("a") for c in name] + [BOS]


def build_dataset(names):
    xs, ys = [], []
    for name in names:
        ids = encode(name)
        for prev, nxt in zip(ids, ids[1:]):
            xs.append(prev)
            ys.append(nxt)
    return np.array(xs), np.array(ys)


def main():
    rng = np.random.default_rng(42)
    names = Path("data/names.txt").read_text().splitlines()
    xs, ys = build_dataset(names)
    N = len(xs)
    print(f"dataset: {N:,} pairs")

    W = Tensor(rng.standard_normal((VOCAB_SIZE, VOCAB_SIZE)) * 0.01)
    lr = 50.0
    epochs = 200

    for epoch in range(epochs):
        W.grad = np.zeros_like(W.data)

        logits = W.gather_cols(xs).transpose()  # (N, V)
        loss = logits.cross_entropy(ys)
        loss.backward()

        W.data -= lr * W.grad

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} | loss {float(loss.data):.4f}")

    print(f"\nfinal loss: {float(loss.data):.4f}")
    print(f"step 2 baseline: 2.4617")

    py_rng = random.Random(42)
    print(f"\n--- 10 samples ---")

    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    for i in range(10):
        out = []
        cur = BOS
        for _ in range(20):
            p = softmax(W.data[:, cur])
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
