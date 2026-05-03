"""Verify the WASM forward pass produces the same logits as bench_numpy.py.

Loads the same weights, runs the same forward pass for the same inputs,
and reports max abs diff. If the WASM result is real, the diff should be
sub-1e-3 (only fp32 rounding differences from add ordering).
"""
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the reference forward exactly as bench_numpy.py defines it.
sys.path.insert(0, str(Path("../benchmark/talos-vs-macbook-m5-pro").resolve()))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bench_numpy_ref",
    "../benchmark/talos-vs-macbook-m5-pro/bench_numpy.py",
)
ref = importlib.util.module_from_spec(spec)
# bench_numpy.py imports `model` so we need that on the path
sys.path.insert(0, "../benchmark/talos-vs-macbook-m5-pro")
spec.loader.exec_module(ref)

print("=" * 60)
print("Correctness check: numpy reference vs WASM")
print("=" * 60)

# Logits at tok=BOS=26, pos=0
K = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
V = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
logits_np = ref.forward(ref.BOS, 0, K, V)

wasm_logits = [
    1.024986, -0.518191, -0.034956, 0.022252, 0.008249, -1.503535,
    -1.421363, -0.843298, -0.685915, 0.461157, 0.558786, 0.009492,
    0.438759, -0.346314, -0.883498, -1.394593, -4.642967, -0.276838,
    0.322638, -0.344579, -2.205366, -1.177411, -1.963754, -2.716714,
    -1.107494, -0.297019, -2.501697,
]
wasm_logits = np.array(wasm_logits, dtype=np.float32)

diff = np.abs(logits_np - wasm_logits)
print(f"\nFirst-token logits (tok=BOS, pos=0):")
print(f"  numpy max abs:  {np.abs(logits_np).max():.6f}")
print(f"  wasm  max abs:  {np.abs(wasm_logits).max():.6f}")
print(f"  max |diff|:     {diff.max():.6f}")
print(f"  mean |diff|:    {diff.mean():.6f}")
print(f"  argmax(numpy):  {logits_np.argmax()}  (logit={logits_np.max():.4f})")
print(f"  argmax(wasm):   {wasm_logits.argmax()}  (logit={wasm_logits.max():.4f})")
print(f"  argmax match:   {logits_np.argmax() == wasm_logits.argmax()}")

# Autoregressive trace: feed 'emma' (BOS, e, m, m, a) and check each step's argmax
K = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
V = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
seq = [26, 4, 12, 12, 0]  # BOS, e, m, m, a
print(f"\nAutoregressive trace for 'emma':")
print(f"  {'pos':>3} {'tok':>3} {'np_max':>8} {'np_arg':>6}  {'wasm_max':>8} {'wasm_arg':>8}  match")

# WASM trace from the browser eval
wasm_trace = [
    (0, 26, 1.0250, 0),
    (1, 4,  2.0051, 11),
    (2, 12, 1.6332, 8),
    (3, 12, 1.8131, 8),
    (4, 0,  1.9175, 13),
]

for p in range(len(seq)):
    logits = ref.forward(seq[p], p, K, V)
    np_max = float(logits.max())
    np_arg = int(logits.argmax())
    _, _, w_max, w_arg = wasm_trace[p]
    match = "OK" if np_arg == w_arg and abs(np_max - w_max) < 1e-2 else "MISMATCH"
    print(f"  {p:>3} {seq[p]:>3} {np_max:>8.4f} {np_arg:>6}  {w_max:>8.4f} {w_arg:>8}  {match}")
