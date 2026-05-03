"""
Tiny autograd library used by Steps 4-6.

Step 3 introduced the idea (Tensor + _backward + topo walk). This file is
the same idea expanded to all the ops we need: matmul, broadcasting add/mul,
tanh/relu, embedding lookup, reshape/transpose, softmax, cross_entropy,
masked_fill, and rms_norm.

A few ops (softmax, cross_entropy, rms_norm) are implemented as primitives
with custom backwards rather than composing from exp/log/sum. That's for
numerical stability and to keep the backward pass fast.
"""
import numpy as np

# macOS Accelerate's matmul spuriously raises FPE flags even on healthy
# inputs. The math is correct; only the status flags are noisy.
np.seterr(invalid="ignore", divide="ignore", over="ignore")


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum out broadcasted axes so grad has shape target_shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(target_shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._children = _children
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    # ---- arithmetic ----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=np.float64))
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=np.float64))
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            gs = out.grad @ other.data.swapaxes(-1, -2)
            go = self.data.swapaxes(-1, -2) @ out.grad
            self.grad += _unbroadcast(gs, self.data.shape)
            other.grad += _unbroadcast(go, other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self + (-other)
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self * (other ** -1.0)
        return self * (1.0 / other)

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar power for now"
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += out.grad * (p * self.data ** (p - 1))

        out._backward = _backward
        return out

    # ---- activations ----
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - t * t)

        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(self.data, 0)
        out = Tensor(out_data, (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += out.grad * e

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    # ---- reductions ----
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for a in sorted(axes):
                    grad = np.expand_dims(grad, a)
            self.grad += np.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- shape ops ----
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if not axes:
            ax = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            ax = tuple(axes[0])
        else:
            ax = axes
        out_data = self.data.transpose(ax) if ax is not None else self.data.T
        out = Tensor(out_data, (self,), "T")

        def _backward():
            if ax is None:
                self.grad += out.grad.T
            else:
                inv = tuple(int(i) for i in np.argsort(ax))
                self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    # ---- indexing ----
    def embedding(self, idx):
        """self has shape (V, D). idx has any shape S. Result has shape S+(D,)."""
        out = Tensor(self.data[idx], (self,), "embedding")

        def _backward():
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        """Where mask is True, replace with `value`. The mask is constant."""
        out_data = np.where(mask, value, self.data)
        out = Tensor(out_data, (self,), "masked_fill")

        def _backward():
            self.grad += np.where(mask, 0.0, out.grad)

        out._backward = _backward
        return out

    # ---- composite ops with custom backward (stability/speed) ----
    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,), "softmax")

        def _backward():
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        """self has shape (..., V). targets is an int array with shape (...,)."""
        z = self.data - self.data.max(axis=-1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        flat_logp = log_probs.reshape(-1, self.data.shape[-1])
        flat_t = np.asarray(targets).reshape(-1)
        N = flat_t.size
        loss_val = -flat_logp[np.arange(N), flat_t].mean()
        out = Tensor(loss_val, (self,), "cross_entropy")

        def _backward():
            probs = np.exp(log_probs)
            grad = probs.reshape(-1, self.data.shape[-1]).copy()
            grad[np.arange(N), flat_t] -= 1.0
            grad /= N
            grad = grad.reshape(self.data.shape)
            self.grad += grad * out.grad

        out._backward = _backward
        return out

    def rms_norm(self, gamma, eps=1e-5):
        """Normalize the last axis by its RMS, then scale by gamma.

        rms = sqrt(mean(x^2) + eps);  out = (x / rms) * gamma
        gamma is a Tensor with shape matching the last axis.
        """
        ms = (self.data ** 2).mean(axis=-1, keepdims=True)
        rms = np.sqrt(ms + eps)
        normed = self.data / rms
        out_data = normed * gamma.data
        out = Tensor(out_data, (self, gamma), "rms_norm")

        def _backward():
            # d(out)/d(gamma) is normed
            gamma.grad += _unbroadcast((out.grad * normed), gamma.data.shape)
            # d(out)/d(x): (gamma/rms) * (out.grad - normed * mean(out.grad * normed))
            d = self.data.shape[-1]
            g = out.grad * gamma.data
            mean_term = (g * normed).sum(axis=-1, keepdims=True) / d
            self.grad += (g - normed * mean_term) / rms

        out._backward = _backward
        return out

    # ---- backward ----
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
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
