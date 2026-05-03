# Handoff: micro-gpt across the abstraction stack

**For a fresh Claude (or human) session picking this up.**

## What this project is

A 4,192-parameter transformer (Karpathy's microGPT) implemented from scratch
in Python and benchmarked across eight substrates: pure Python, NumPy, MLX-CPU,
MLX-GPU, TALOS-V2 FPGA (reference), hand-written C+NEON, and WebAssembly running
in Chrome on the user's Apple M4 Pro (24 GB).

The repo is **public** and **live on GitHub Pages**:

- Source: https://github.com/NakliTechie/micro-gpt
- Live demo: https://naklitechie.github.io/micro-gpt/wasm/
- Report: https://naklitechie.github.io/micro-gpt/report/
- Landing: https://naklitechie.github.io/micro-gpt/

Working directory: `/Users/chiragpatnaik/Code/micro-gpt`. User's GitHub handle:
`NakliTechie`. User's email: `chirag.patnaik@gmail.com`.

## Headline result

**WASM in regular Chrome on M4 Pro 24GB hits 1,341,206 ± 2,445 tok/sec** — 25.30×
the TALOS-V2 FPGA reference (53,000 tok/sec at 56 MHz), ~35% of LUT-optimized
native C+NEON (3,820,760 tok/sec). All numbers measured locally.

A second WASM number appears in the report: 2,038,131 tok/sec in
Electron-embedded Chromium on the same M4 Pro hardware. Same `.wasm` binary,
~50% delta from the V8 build alone — itself a finding worth keeping.

## Status

- ✅ Educational ladder complete (`ladder/step1..step6`, with autograd in
  `tensor.py`). Step 6 trains to NLL 2.21 train / 2.20 val on a deduped
  90/10 unique-name split with an `assert isdisjoint` guard.
- ✅ Quantization study complete (`quant/quant_study.py`). Per-tensor int8
  shows no measurable degradation on the 500-name slice.
- ✅ Benchmark replicated on M4 Pro 24GB (`benchmark/results.md`,
  `benchmark/talos-vs-macbook-m5-pro/` cloned but gitignored).
- ✅ WASM demo built and deployed (`wasm/`, live on Pages).
- ✅ Live verification: `python3 wasm/verify_against_numpy.py` from repo root —
  spawns `node wasm/dump_logits.js`, loads the built `.wasm` fresh, compares
  logits element-wise against NumPy reference (max |diff| ~1e-6).
- ✅ Two independent skeptical reviews passed: Codex (gpt-5.5) and a Claude
  sub-agent both audited and the final codex run returned PUBLISH.
- ✅ Two announcement tweets drafted (see this file's "Tweets" section
  below — user copied them out manually).
- ⏳ User has ordered three microcontrollers (Pi Pico 2, ESP32-S3, ESP32-P4)
  for a future session.

## Pending work

See `pending.md` for the full list. Order of attack:

1. **MLX `mx.compile` experiment (1 evening, next).** Does fusing the forward
   pass into a single MLX kernel let MLX-GPU beat eager mode? Either result is
   informative.
2. Microcontroller deployments when the three boards arrive.
3. CoreML / Apple Neural Engine.
4. Hand-rolled Metal compute shader (fused kernel + batched streams).
5. Raspberry Pi 5 (drop-in C+NEON).

## Things a new session needs to know

### Tooling

- **Codex CLI** at `/Users/chiragpatnaik/.nvm/versions/node/v22.18.0/bin/codex`
  v0.128.0. Use for skeptical second-opinion code reviews. Default model
  `gpt-5.5` works. Run with `codex exec --skip-git-repo-check --sandbox
  read-only "<prompt>"`. **Don't pipe its output through `tail -N`** — that
  blocks output until exit. Use `> /tmp/codex.log 2>&1 &` and poll.
- **Qwen CLI** authenticated via qwen-oauth, but the free tier was sunset
  2026-04-15. Non-interactive `--prompt` mode rejects qwen-oauth in v0.15.6.
  **Skip qwen** unless the user has switched to OpenRouter/Fireworks/Coding Plan.
- **Gemini CLI** authenticated. Hits Google free-tier 429s aggressively. May
  recover after cooldown. Use `-m gemini-2.5-flash` for higher quota.
- **Claude sub-agents** (general-purpose) for parallel reviews work cleanly
  and are the most reliable second-eye option.

### Caveats baked into the published artifact

- **C+NEON vs WASM is not strict apples-to-apples.** The upstream
  `bench_c.c` precomputes `(token, pos)` LUTs for the model's front half
  *outside* its timed loop; the WASM port computes that work every step.
  Documented in README, `benchmark/results.md`, `report/index.html`,
  `wasm/index.html`. Don't accidentally re-frame as same-workload.
- **Step 6 model ≠ TALOS reference.** Step 6 has learnable RMSNorm gains
  + final RMSNorm = 4,240 params; TALOS is parameter-free RMSNorm + no
  final norm = 4,192 params. Same architecture family, different weights
  and parameter count.
- **Quant claim is "no measurable degradation on 500-name slice"** —
  do not strengthen to "statistically lossless" without a held-out CI.
- **NLL ~2.2 floor is a model-capacity floor at ~4K params**, not a
  dataset entropy floor. Bigger models go below 2.0 on this corpus.

### Rebuild flow

If editing `wasm/microgpt_inf.c`:

```sh
cd wasm && EMSDK_PYTHON=/opt/homebrew/bin/python3.13 ./build.sh
# from repo root:
python3 wasm/verify_against_numpy.py
```

The `EMSDK_PYTHON` override is needed because emscripten 5.0.7 requires
Python 3.10+ but the system Python is 3.9.6.

If you change `microgpt_inf.c`, the verifier only catches regressions
*after* `./build.sh` runs (intentional — keeps emscripten optional for
verification-only users).

### File map

```
README.md                # v1.0 narrative, with try-it-now CTA
NOTICE.md                # provenance of bundled TALOS-derived weights
HANDOFF.md               # this file
index.html               # Pages landing page
pending.md               # the list of remaining experiments
.claude/launch.json      # preview server config (port 8765, serves wasm/)
.gitignore

ladder/                  # 6-step educational walk
  tensor.py              # autograd
  step1_bigram.py..step6_gpt_multi_head.py
  step6_weights.npz      # our trained weights

quant/quant_study.py     # int8/4/2 per-tensor symmetric study

wasm/
  microgpt_inf.c         # the inference forward pass + benchmark()
  microgpt_inf.js, .wasm # built artifacts (committed for Pages)
  weights.bin            # 16,768 bytes of fp32, derived from TALOS
  index.html             # live demo UI
  dump_logits.js         # Node-side WASM driver
  verify_against_numpy.py# live verifier (uses dump_logits.js)
  build.sh
  bench_runs.txt         # raw 10-run benchmark data

benchmark/
  results.md             # M4 Pro 24GB benchmark + cProfile breakdown
  talos-vs-macbook-m5-pro/  # gitignored upstream clone

report/index.html        # interactive Plotly charts
```

### Two slightly-quirky details

1. **macOS Accelerate / NumPy quirk:** `tensor.py` does
   `np.seterr(invalid="ignore", divide="ignore", over="ignore")` because
   Accelerate's matmul spuriously raises FPE flags on healthy inputs.
   Don't remove the seterr.
2. **`-sENVIRONMENT=web,node` in `wasm/build.sh`** is *required* for the
   verifier to load the wasm under Node. Don't drop the `node`.

## Tweets the user posted (manually) when this session ended

Tweet 1 (with chart screenshot):
> Did this 👉 naklitechie.github.io/micro-gpt/wasm/
>
> @karpathy's microGPT running entirely in your browser as WebAssembly —
> on-brand with the other browser experiments I keep doing.
> 1.34M tok/sec on M4 Pro, 25× the FPGA TALOS-V2 runs the same model on.

Tweet 2:
> How we got here:
> 1. read v2.talos.wtf — microGPT in Verilog at 56 MHz
> 2. ran the same model on M4 Pro: NumPy and MLX *lose* to that FPGA
>    (dispatch overhead)
> 3. C+NEON crushes everything
> 4. ported to WASM — 25× the FPGA in any browser
>
> Repo: github.com/NakliTechie/micro-gpt

## How to pick up

If the user comes back saying "let's do the MLX compile experiment next":
go to `pending.md` item 5, write `bench_mlx_compiled.py` with `@mx.compile`,
warm up, benchmark, compare to the eager-MLX 1,865 / 3,873 numbers, write
up.

If the user comes back saying "the boards arrived": go to `pending.md`
item 2. Start with the Pico 2 (forces Q4.12 — closest analogue to TALOS).
The C source in `wasm/microgpt_inf.c` is mostly portable; the main change
is fixed-point math instead of fp32.

If the user comes back saying "something is wrong on the live demo":
the demo is at `wasm/index.html` on the `main` branch; Pages rebuilds in
~10s after `git push`. The first thing to check is whether
`microgpt_inf.{js,wasm}` and `weights.bin` are present (they should be
committed; see `.gitignore` for the disabled lines).

For anything else, read `README.md` first — it's the v1.0 description of
what's here. Then `pending.md`. Then this file.
