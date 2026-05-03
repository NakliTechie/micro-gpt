# micro-gpt across the abstraction stack

> **▶ [Try the live WASM demo](https://NakliTechie.github.io/micro-gpt/wasm/)** &nbsp;·&nbsp;
> **📊 [Full interactive report](https://NakliTechie.github.io/micro-gpt/report/)** &nbsp;·&nbsp;
> **🏠 [Overview](https://NakliTechie.github.io/micro-gpt/)**
>
> *(GitHub Pages must be enabled on this repository for those links to resolve. See
> "Hosting on GitHub Pages" at the end of this README.)*

A 4,192-parameter transformer (Karpathy's microGPT, character-level, names dataset),
implemented from scratch in Python and benchmarked across eight substrates:
pure Python, NumPy, MLX-CPU, MLX-GPU, TALOS-V2 FPGA (reference, 56 MHz Cyclone V),
hand-written C+NEON, and WebAssembly running in Chrome on the M4 Pro.

## What this repo shows

- **Framework dispatch dominates the math at this model scale.** NumPy spends
  ~96% of wall-clock on reduce/dispatch/typecheck, only ~4% in the actual
  matmul kernel. MLX-GPU loses to *pure Python* because the GPU launch
  overhead is the wrong shape for 4K MACs per token.

- **There is a compiled-vs-runtime cliff.** Python / NumPy / MLX all sit
  below the FPGA. C+NEON / WASM / TALOS all crush it. The cliff is about
  three orders of magnitude.

- **WebAssembly hits ~1.34M tok/sec in Chrome on M4 Pro — 25× the FPGA, ~35%
  of a LUT-optimized native C+NEON harness.** The browser is a viable
  inference target for tiny transformers.

- **The model has at most 8 effective bits of precision.** Per-tensor int8
  quantization shows no measurable degradation on a 500-name slice. Q4.12
  (16-bit, what TALOS uses) carries headroom unused by this model's weight
  distribution — chosen for hardware reasons, not accuracy.

- **Same WASM binary, different V8 builds, ~50% throughput delta.** Regular
  Chrome 145 measures ~1.34M tok/sec on M4 Pro; Electron-embedded Chromium
  146 measures ~2.04M tok/sec on the same hardware. WebAssembly performance
  isn't a single number per machine — it's a band per (binary, host) pair.

## Project layout

```
ladder/                  # 6-step educational walk: bigram → MLP → GPT
  step1_bigram.py        # counting + sampling, no learning (NLL 2.45)
  step2_neural_bigram.py # manual gradients (NLL 2.46)
  step3_autograd.py      # graph-based backprop (NLL 2.46)
  step4_mlp.py           # embeddings + 3-char context + tanh (NLL 2.20)
  step5_gpt_single_head.py  # self-attention + RMSNorm (NLL 2.29)
  step6_gpt_multi_head.py   # multi-head + Adam (train 2.21 / val 2.20,
                            #  90/10 split over deduped unique names)
  tensor.py              # autograd library used by steps 4-6
  step6_weights.npz      # trained weights (4,240 params)

quant/
  quant_study.py         # Q4.12, Q3.13, int8/4/2 per-tensor symmetric
                         #  quantization study on TALOS-trained weights

wasm/                    # WebAssembly inference target
  microgpt_inf.c         # ~150 lines, the entire forward pass
  build.sh               # emcc -O3 -msimd128 -ffast-math
  index.html             # browser harness, generates names + benchmark
  dump_logits.js         # Node-side WASM driver used by the verifier
  verify_against_numpy.py# loads the built .wasm via Node, compares
                         #  logits element-wise against the NumPy reference
  bench_runs.txt         # raw benchmark data, both browsers

benchmark/
  results.md             # full benchmark + profiling notes (M4 Pro)

report/
  index.html             # interactive Plotly charts of every result
                         # open directly in a browser

pending.md               # planned follow-up substrates
```

## Reproducing

### 1. The educational ladder (~5 min)

```sh
mkdir -p data && curl -sL \
    https://raw.githubusercontent.com/karpathy/makemore/master/names.txt \
    -o data/names.txt

python3 ladder/step1_bigram.py
python3 ladder/step2_neural_bigram.py
python3 ladder/step3_autograd.py
python3 ladder/step4_mlp.py
python3 ladder/step5_gpt_single_head.py
python3 ladder/step6_gpt_multi_head.py    # writes step6_weights.npz
```

Each step prints loss progression and 10 generated samples. Step 6 reports
a real held-out NLL on a deduped 90/10 train/val split (`assert
disjoint`). The training corpus has duplicates (32,033 rows → 29,494 unique
names), so deduping before splitting matters; a row-level split would leak
~14% of val names into train.

### 2. The benchmark (M4 Pro, single-stream + multi-stream)

```sh
mkdir -p benchmark && cd benchmark
git clone https://github.com/itsrealranky/talos-vs-macbook-m5-pro.git
cd talos-vs-macbook-m5-pro
./run.sh
./bench_c --threads 14 5000000 200000     # aggregate
```

The fork is by Alex Cheema (original) and Ranky (M5 Pro tuning); the trained
weights inside originate from TALOS-V2 by Luthira Abeykoon (no license, so
they aren't redistributed in this repo — `download.sh` fetches them fresh
from upstream).

### 3. The quantization study

`quant/quant_study.py` requires the TALOS-trained weights from upstream.
Run section 2 (or just `./download.sh` inside the cloned fork) so
`benchmark/talos-vs-macbook-m5-pro/assets/weights_only.npy` exists, then:

```sh
python3 quant/quant_study.py
```

### 4. The WebAssembly demo and verification

```sh
cd wasm
./build.sh         # needs emscripten (brew install emscripten)
python3 -m http.server 8765
# open http://localhost:8765 in regular Chrome
# click "Generate 20 names" or "Benchmark (100K tokens)"
```

To verify the WASM forward pass element-wise against the NumPy reference,
**from the repo root**:

```sh
python3 wasm/verify_against_numpy.py
```

This spawns `node wasm/dump_logits.js`, which loads the *built*
`microgpt_inf.wasm` and `weights.bin`, dumps logits at fixed inputs, and
compares against the reference. Max |diff| ≈ 10⁻⁶ on M4 Pro; argmax
identical at every autoregressive position checked. Requires `node` on
PATH and the upstream fork's assets from section 2. The script tests the
built artifact, so after editing `microgpt_inf.c` you must re-run
`./build.sh` first; emscripten is intentionally not made a hard dependency
of the verifier so verification-only users don't need it.

### 5. The full report

```sh
open report/index.html
```

Static HTML, no server needed. Six interactive Plotly charts.

## Single-stream throughput on Apple M4 Pro

| implementation | tok/sec | vs FPGA |
|---|---:|---:|
| MLX GPU | 1,865 | 0.04× |
| MLX CPU | 3,873 | 0.07× |
| pure Python | 4,332 | 0.08× |
| NumPy fp32 | 24,223 | 0.46× |
| **TALOS-V2 (FPGA, 56 MHz)** | **53,000** | **1.00×** |
| WASM (Chrome 145) ★★ | 1,341,206 ± 2,445 | 25.30× |
| WASM (Chromium 146 / Electron 41) ★★ | 2,038,131 ± 20,000 | 38.46× |
| C+NEON Q4.12 ★ | 2,191,219 | 41.34× |
| C+NEON fp32 ★ | 3,820,760 | 72.09× |
| C+NEON ×14 streams (aggregate) ★ | 32,894,149 | 620.6× |

★ The C+NEON harness from upstream `talos-vs-macbook-m5-pro` precomputes
`(token, pos)` LUTs for the model's front half *outside* the timed loop.
The WASM port computes that work inside the timed loop. The comparison is
meaningful but not strict apples-to-apples; treat it as "browser WASM vs
LUT-optimized native." See [benchmark/results.md](benchmark/results.md).

★★ The WASM number depends on the V8 build. Both rows are the same `.wasm`
binary on the same M4 Pro; only the host runtime differs. See
[wasm/bench_runs.txt](wasm/bench_runs.txt) for raw runs.

## Verification posture

- **WASM forward pass:** logits agree with the NumPy reference to max
  |diff| ≈ 10⁻⁶ (live verification via `node`).
- **Step 6 train/val:** 90/10 split over *unique* names; assertion
  enforces disjointness; train NLL 2.21, val NLL 2.20.
- **Quantization claim:** "No measurable degradation on a 500-name slice"
  (training-corpus eval, no held-out CI).
- **Independent review:** the repo was reviewed by Codex and an
  independent Claude sub-agent before publication; review findings and
  fixes are in the commit history (search `git log --grep=codex`).

## Hosting on GitHub Pages

The `index.html` at the repo root is a small landing page that links to
the live WASM demo (`wasm/index.html`) and the interactive report
(`report/index.html`). The committed WASM build artifacts
(`microgpt_inf.js`, `microgpt_inf.wasm`, `weights.bin`) make the demo
self-contained — no build step needed for visitors.

**To enable Pages on this repo:**

```sh
gh api repos/:owner/:repo/pages -X POST \
  -f 'source[branch]=main' -f 'source[path]=/'
```

Or via the web UI: Settings → Pages → Source: `main` branch, `/` (root).

Pages on **private** repositories requires GitHub Pro / Team /
Enterprise. On the free tier, the repo must be public. After enabling,
the demo lives at:

- `https://<user>.github.io/<repo>/` — the landing page
- `https://<user>.github.io/<repo>/wasm/` — the live demo
- `https://<user>.github.io/<repo>/report/` — the report

See [NOTICE.md](NOTICE.md) for the provenance of bundled weights.

## Influences

- [Karpathy's microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the model.
- [TALOS-V2](https://github.com/Luthiraa/TALOS-V2) — the FPGA implementation that started it all.
- [v2.talos.wtf](https://v2.talos.wtf/) — the design write-up.
- [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c) — pure-C training + inference.
- [microgpt-java](https://github.com/ani03sha/microgpt-java) — the educational ladder structure.
- [talos-vs-macbook-m5-pro](https://github.com/itsrealranky/talos-vs-macbook-m5-pro) — the benchmark harness.
