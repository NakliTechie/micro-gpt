"""
Step 1: Statistical Bigram Model.

For each character pair (prev, next), count how often it appears.
P(next | prev) = count(prev, next) / sum_n count(prev, n).
Add-one (Laplace) smoothing so unseen pairs are non-zero.
Generate by sampling from P(. | current) until BOS reappears.

This is pure counting -- no learning, no gradients. The NLL it achieves
is the bar every later step in the ladder has to beat.
"""
import math
import random
from pathlib import Path

VOCAB_SIZE = 27
BOS = 26  # also used as end-of-name marker


def char_to_id(c: str) -> int:
    return ord(c) - ord("a")


def id_to_char(i: int) -> str:
    return "." if i == BOS else chr(i + ord("a"))


def encode(name: str) -> list[int]:
    return [BOS] + [char_to_id(c) for c in name] + [BOS]


def build_counts(names: list[str]) -> list[list[int]]:
    counts = [[0] * VOCAB_SIZE for _ in range(VOCAB_SIZE)]
    for name in names:
        ids = encode(name)
        for prev, nxt in zip(ids, ids[1:]):
            counts[prev][nxt] += 1
    return counts


def counts_to_probs(counts: list[list[int]], smoothing: int = 1) -> list[list[float]]:
    probs = []
    for row in counts:
        total = sum(row) + smoothing * VOCAB_SIZE
        probs.append([(c + smoothing) / total for c in row])
    return probs


def sample(dist: list[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(dist):
        cum += p
        if r < cum:
            return i
    return len(dist) - 1


def generate(probs: list[list[float]], rng: random.Random, max_len: int = 20) -> str:
    out: list[str] = []
    cur = BOS
    for _ in range(max_len):
        cur = sample(probs[cur], rng)
        if cur == BOS:
            break
        out.append(id_to_char(cur))
    return "".join(out)


def avg_nll(probs: list[list[float]], names: list[str]) -> float:
    total_ll = 0.0
    n = 0
    for name in names:
        ids = encode(name)
        for prev, nxt in zip(ids, ids[1:]):
            total_ll += math.log(probs[prev][nxt])
            n += 1
    return -total_ll / n


def top_k_transitions(probs: list[list[float]], from_char: str, k: int = 5):
    row = probs[BOS] if from_char == "." else probs[char_to_id(from_char)]
    ranked = sorted(enumerate(row), key=lambda x: -x[1])
    return [(id_to_char(i), p) for i, p in ranked[:k]]


def main():
    rng = random.Random(42)
    names = Path("data/names.txt").read_text().splitlines()
    print(f"loaded {len(names):,} names, e.g. {names[:5]}")

    counts = build_counts(names)
    probs = counts_to_probs(counts, smoothing=1)

    print(f"\ntop-5 first-character transitions (from BOS):")
    for c, p in top_k_transitions(probs, ".", 5):
        print(f"  '{c}' with p={p:.3f}")
    print(f"\ntop-5 transitions from 'e':")
    for c, p in top_k_transitions(probs, "e", 5):
        print(f"  '{c}' with p={p:.3f}")

    nll = avg_nll(probs, names)
    print(f"\naverage NLL on training data: {nll:.4f}")
    print(f"uniform-random baseline:      {math.log(VOCAB_SIZE):.4f}")

    print(f"\n--- 10 generated samples ---")
    for i in range(10):
        print(f"sample {i+1:2d}: {generate(probs, rng)}")


if __name__ == "__main__":
    main()
