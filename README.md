# micro-gpt across the abstraction stack

A 4,192-parameter transformer (Karpathy's microGPT, character-level, names dataset),
implemented from scratch in Python and benchmarked across **eight** substrates:
pure Python, NumPy, MLX-CPU, MLX-GPU, TALOS-V2 FPGA (reference, 56 MHz Cyclone V),
hand-written C+NEON, and WebAssembly running in Chrome on the M4 Pro.

The interesting findings:

- **Framework dispatch dominates the math at this model scale.** NumPy spends
  ~96% of wall-clock on reduce/dispatch/typecheck, only ~4% in the actual
  matmul kernel. MLX-GPU loses to *pure Python* because the GPU launch
  overhead is the wrong shape for 4K MACs per token.
- **The compiled-vs-runtime cliff is real.** Python / NumPy / MLX all sit
  below the FPGA. C+NEON / WASM / TALOS all crush the FPGA. The cliff is
  about three orders of magnitude.
- **WebAssembly hits ~2.04M tok/sec on M4 Pro &mdash; 38× the FPGA, ~53%
  of a LUT-optimized native C+NEON harness.** The browser is a viable
  inference target for tiny transformers. (Caveat: the native C+NEON harness
  precomputes the model's front half into lookup tables outside the timed
  loop; WASM does that work inside the loop. The comparison is meaningful
  but not strict apples-to-apples — see [benchmark/results.md](benchmark/results.md).)
- **The model has at most 8 effective bits of precision.** Per-tensor int8
  quantization shows no measurable degradation on a 500-name slice
  (training-corpus eval, no held-out set). Q4.12 (16-bit, what TALOS uses)
  carries headroom unused by this model's weight distribution &mdash; chosen
  for hardware reasons, not accuracy.

## Project layout

```
ladder/                  # 6-step educational walk: bigram → MLP → GPT
  step1_bigram.py        # counting + sampling, no learning (NLL 2.45)
  step2_neural_bigram.py # manual gradients (NLL 2.46)
  step3_autograd.py      # graph-based backprop (NLL 2.46)
  step4_mlp.py           # embeddings + 3-char context + tanh (NLL 2.20)
  step5_gpt_single_head.py  # self-attention + RMSNorm (NLL 2.29)
  step6_gpt_multi_head.py   # multi-head + Adam (train 2.21 / val 2.20, deduped 90/10)
  tensor.py              # autograd library used by steps 4-6
  step6_weights.npz      # our trained weights (4,240 params)

quant/
  quant_study.py         # Q4.12, Q3.13, int8/4/2 per-tensor symmetric
                         #  quantization study on TALOS-trained weights

wasm/                    # WebAssembly inference target
  microgpt_inf.c         # ~150 lines, the entire forward pass
  build.sh               # emcc -O3 -msimd128 -ffast-math
  index.html             # browser harness, generates names + benchmark
  verify_against_numpy.py# sanity check WASM logits == numpy logits

benchmark/
  results.md             # full benchmark + profiling notes (M4 Pro)

report/
  index.html             # interactive Plotly charts of every result
                         # open directly in a browser

pending.md               # planned follow-up substrates
```

## Reproducing

### 1. The educational ladder (~5 minutes total)

```sh
# Get the dataset (32K names, ~220 KB)
mkdir -p data && curl -sL \
    https://raw.githubusercontent.com/karpathy/makemore/master/names.txt \
    -o data/names.txt

# Each step prints loss progression and 10 generated samples
python3 ladder/step1_bigram.py
python3 ladder/step2_neural_bigram.py
python3 ladder/step3_autograd.py
python3 ladder/step4_mlp.py
python3 ladder/step5_gpt_single_head.py
python3 ladder/step6_gpt_multi_head.py    # writes step6_weights.npz
```

### 2. The benchmark (M4 Pro single-stream + multi-stream)

```sh
mkdir -p benchmark && cd benchmark
git clone https://github.com/itsrealranky/talos-vs-macbook-m5-pro.git
cd talos-vs-macbook-m5-pro
./run.sh
./bench_c --threads 14 5000000 200000     # aggregate
```

The fork is by Alex Cheema (original) and Ranky (M5 Pro tuning); the trained
weights inside originate from TALOS-V2 by Luthira Abeykoon (no license, do not
redistribute).

### 3. The quantization study

`quant/quant_study.py` requires the TALOS-trained weights from the upstream
fork's assets. Run section 2 (clone + `./run.sh` or just `./download.sh`)
first so `benchmark/talos-vs-macbook-m5-pro/assets/weights_only.npy` exists.

```sh
python3 quant/quant_study.py
```

### 4. The WebAssembly demo

```sh
cd wasm
./build.sh         # needs emscripten installed (brew install emscripten)
python3 -m http.server 8765
# open http://localhost:8765 — generates names, has a benchmark button
```

To verify the WASM logits match the NumPy reference within fp32 rounding,
**from the repo root**:

```sh
python3 wasm/verify_against_numpy.py
```

This actually loads the *built* `wasm/microgpt_inf.wasm` via Node and compares
logits element-wise against the NumPy reference forward pass. (Requires `node`
on PATH and the upstream fork's assets from section 2.) Note: the script
verifies the built artifact, so to catch a regression in `microgpt_inf.c` you
must `./build.sh` first; the script intentionally does not rebuild for you,
to keep emscripten an optional dependency for verification-only users.

### 5. The full report

```sh
open report/index.html      # static HTML, no server needed
```

## Single-stream throughput on Apple M4 Pro

| implementation | tok/sec | vs FPGA |
|---|---:|---:|
| MLX GPU | 1,865 | 0.04× |
| MLX CPU | 3,873 | 0.07× |
| pure Python | 4,332 | 0.08× |
| NumPy fp32 | 24,223 | 0.46× |
| **TALOS-V2 (FPGA, 56 MHz)** | **53,000** | **1.00×** |
| WASM (Chromium) ★★ | 2,038,131 ± 20,000 | 38.46× |
| C+NEON Q4.12 ★ | 2,191,219 | 41.34× |
| C+NEON fp32 ★ | 3,820,760 | 72.09× |
| C+NEON ×14 streams (aggregate) ★ | 32,894,149 | 620.6× |

★ The C+NEON harness from upstream `talos-vs-macbook-m5-pro` precomputes
`(token, pos)` LUTs for the model's front half *outside* the timed loop
(see [benchmark/results.md](benchmark/results.md)). The WASM port computes
that work inside the timed loop. The comparison is meaningful but not strict
apples-to-apples; treat it as "browser WASM vs LUT-optimized native."

★★ Mean of 10 runs of 100,000 tokens each, after 20,000-token warmup, two
separate page sessions. CV across runs ≈ 1.4%. Raw data in
[wasm/bench_runs.txt](wasm/bench_runs.txt).

`python3 wasm/verify_against_numpy.py` (from the repo root) compares the live
WASM logits element-wise against the NumPy reference: max |diff| ≈ 10⁻⁶,
argmax identical at every autoregressive position checked. The verification
script loads the **built** `microgpt_inf.wasm` (run `./build.sh` after any
edit to `microgpt_inf.c` to bring the artifact up-to-date before verifying).

## Influences

- [Karpathy's microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the model.
- [TALOS-V2](https://github.com/Luthiraa/TALOS-V2) — the FPGA implementation that started it all.
- [v2.talos.wtf](https://v2.talos.wtf) — the write-up on hardware design choices.
- [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c) — pure-C training + inference.
- [microgpt-java](https://github.com/ani03sha/microgpt-java) — the educational ladder structure.
- [talos-vs-macbook-m5-pro](https://github.com/itsrealranky/talos-vs-macbook-m5-pro) — the benchmark harness.
