"""Live verification: actually run the WASM module under Node, dump logits,
compare element-wise against the NumPy reference forward pass.

If the WASM is silently broken, this script will fail loudly. The earlier
version of this file compared against pasted literals; that has been replaced
with `node dump_logits.js` which loads `microgpt_inf.wasm` and `weights.bin`
fresh on every run.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WASM_DIR = Path(__file__).resolve().parent
REPO_ROOT = WASM_DIR.parent
ASSETS = REPO_ROOT / "benchmark/talos-vs-macbook-m5-pro/assets"

if not ASSETS.exists():
    print("[error] benchmark/talos-vs-macbook-m5-pro/assets/ not found.")
    print("[error] Clone the upstream fork and run ./download.sh first:")
    print("[error]   cd benchmark && git clone https://github.com/itsrealranky/talos-vs-macbook-m5-pro.git")
    print("[error]   cd talos-vs-macbook-m5-pro && ./download.sh")
    sys.exit(2)

# Load the reference forward exactly as bench_numpy.py defines it
sys.path.insert(0, str(ASSETS.parent))  # for `from model import ...`
import importlib.util

spec = importlib.util.spec_from_file_location(
    "bench_numpy_ref", ASSETS.parent / "bench_numpy.py",
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

# Live-call the WASM via Node
print("[info] running node wasm/dump_logits.js (live WASM, fresh load)...")
proc = subprocess.run(
    ["node", "dump_logits.js"],
    cwd=WASM_DIR,
    capture_output=True,
    text=True,
    check=False,
)
if proc.returncode != 0:
    print(f"[error] node failed (exit {proc.returncode})")
    print(proc.stderr)
    sys.exit(1)

wasm_data = json.loads(proc.stdout)
wasm_first = np.array(wasm_data["first_logits"], dtype=np.float32)

print("=" * 64)
print("Correctness: NumPy reference vs LIVE WASM (loaded by Node this run)")
print("=" * 64)

# 1. Logits at tok=BOS=26, pos=0
K = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
V = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
logits_np = ref.forward(ref.BOS, 0, K, V)

diff = np.abs(logits_np - wasm_first)
print(f"\nFirst-token logits (tok=BOS=26, pos=0):")
print(f"  numpy max abs:  {np.abs(logits_np).max():.6f}")
print(f"  wasm  max abs:  {np.abs(wasm_first).max():.6f}")
print(f"  max |diff|:     {diff.max():.6f}")
print(f"  mean |diff|:    {diff.mean():.6f}")
print(f"  argmax(numpy):  {logits_np.argmax()}  (logit={logits_np.max():.4f})")
print(f"  argmax(wasm):   {wasm_first.argmax()}  (logit={wasm_first.max():.4f})")
match = logits_np.argmax() == wasm_first.argmax()
print(f"  argmax match:   {match}")
if not match or diff.max() > 1e-3:
    print("[FAIL] mismatch beyond fp32 rounding tolerance")
    sys.exit(1)

# 2. Autoregressive trace: feed 'emma'
K = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
V = np.zeros((ref.BLOCK_SIZE, ref.N_EMBD), dtype=np.float32)
print(f"\nAutoregressive trace for 'emma':")
print(f"  {'pos':>3} {'tok':>3} {'np_max':>8} {'np_arg':>6}  {'wasm_max':>8} {'wasm_arg':>8}  status")
all_ok = True
for entry in wasm_data["autoregressive_trace"]:
    pos, tok = entry["pos"], entry["tok"]
    wasm_logits = np.array(entry["logits"], dtype=np.float32)
    np_logits = ref.forward(tok, pos, K, V)
    np_arg = int(np_logits.argmax())
    w_arg = int(wasm_logits.argmax())
    np_max = float(np_logits.max())
    w_max = float(wasm_logits.max())
    diff_pos = float(np.abs(np_logits - wasm_logits).max())
    ok = (np_arg == w_arg) and (diff_pos < 1e-3)
    all_ok = all_ok and ok
    print(f"  {pos:>3} {tok:>3} {np_max:>8.4f} {np_arg:>6}  {w_max:>8.4f} {w_arg:>8}  {'OK' if ok else 'FAIL'} (max|d|={diff_pos:.6f})")

# 3. Performance stability
print(f"\nWASM throughput (5 runs of 100K tokens, after 20K warmup):")
runs = wasm_data["tps_runs"]
mean = sum(runs) / len(runs)
std = (sum((r - mean) ** 2 for r in runs) / len(runs)) ** 0.5
print(f"  runs:       {[round(r) for r in runs]}")
print(f"  mean:       {round(mean):,} tok/sec")
print(f"  std:        {round(std):,} tok/sec")
print(f"  CV:         {std / mean * 100:.2f}%")

print(f"\n{'PASS' if all_ok else 'FAIL'}: live WASM matches NumPy reference within fp32 rounding.")
sys.exit(0 if all_ok else 1)
