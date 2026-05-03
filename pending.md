# Pending experiments

Replication targets for the same 4,192-parameter microGPT model across additional substrates.
Each one closes a specific data point in the "where does time actually go" map.

## 1. WebAssembly  &mdash; DONE (live at https://naklitechie.github.io/micro-gpt/wasm/)

**Result.** **1,341,206 ± 2,445 tok/sec** in regular Chrome on M4 Pro 24GB &mdash; **25.30&times; the FPGA**,
**~35% of native C+NEON** (3.82M). Surprise finding: the same `.wasm` measures **2,038,131 tok/sec** in
Electron-embedded Chromium on the same machine &mdash; ~50% delta from the V8 build alone.

**What this told us.** The "runtime vs compiled" cliff hypothesis was correct: WASM sits firmly on the
*fast* side of the cliff, three orders of magnitude above NumPy and MLX. WebAssembly throughput is a band
per (binary, host) pair, not a single number per machine.

**Files.** `wasm/microgpt_inf.c`, `wasm/index.html`, `wasm/build.sh`, `wasm/dump_logits.js`,
`wasm/verify_against_numpy.py`, `wasm/bench_runs.txt`. The verifier loads the built `.wasm` via Node and
compares logits element-wise against the NumPy reference (max |diff| ~1e-6).

## 2. Microcontroller deployment  (boards ordered)

**Goal.** Run the same model on three different MCU architectures.

**Boards on order.**
- Raspberry Pi Pico 2 (RP2350) — dual Cortex-M33 + dual RISC-V, no FPU by default. Forces Q4.12.
- ESP32-S3-DevKitC — Xtensa LX7 dual-core, vector AI instructions.
- ESP32-P4 — RISC-V Hi-Fi 5, 400 MHz, FPU + AI vector extensions.

**Why.** Each lands at a different precision/architecture point:
- Pico 2: no FPU at all → Q4.12 is the only option, exactly the FPGA's constraint on a $5 chip.
- ESP32-S3: tight-budget SIMD with mature inference toolchain.
- ESP32-P4: "what if the microcontroller was AI-aware?" data point.

**Build path.** Port `bench_c.c` to fixed-point Q4.12 only (no fp32 fallback) so it compiles for the Pico 2.
For ESP32-S3 / P4 build through ESP-IDF using their vector intrinsics for the inner loop.

**Estimated effort.** 1 weekend once boards arrive.

## 3. Apple Neural Engine via CoreML

**Goal.** Run the trained model on the ANE (the third major accelerator class on M-series silicon, after CPU and GPU).

**Why.** APIs are public via CoreML — the *private* `ANE.framework` is closed but `coremltools` reaches the same hardware. Honest expectation: ANE *may* also lose to C+NEON because it has its own dispatch overhead, but the *shape* of that overhead is different from GPU launch. Worth knowing.

**Build path.**
1. Implement the model in PyTorch (or load step6_weights.npz into a torch module).
2. `coremltools.convert(..., compute_units=ct.ComputeUnit.CPU_AND_NE)` → `.mlpackage`.
3. Benchmark via `MLModel.predict()` in Python or Swift. Verify ANE dispatch with `xcrun coremlc analyze`.

**Estimated effort.** 1 evening.

## 4. Hand-rolled Metal compute shader

**Goal.** Beat the per-op-launch penalty that killed MLX-GPU (1,865 tok/sec on the M4).

**Why.** The MLX failure is "~15 kernel launches per token × 70 µs each = 1 ms/token". A *single fused* Metal kernel that runs the whole forward pass should drop per-token latency to ~10-20 µs (50-100K tok/sec single-stream). The bigger win is **batched** — run hundreds of independent streams in one kernel launch. Aggregate could plausibly hit 50-100M tok/sec, beating the C+NEON 14-thread aggregate.

This is the most ambitious item on the list and the one that directly answers "can we beat C+NEON anywhere on this Mac?"

**Build path.**
1. Single fused-forward kernel in Metal Shading Language (one threadgroup per stream, one thread per attention head).
2. Host-side launches the kernel, reads back logits, samples on CPU.
3. Then convert to batched: N streams in parallel, one launch per N tokens.

**Estimated effort.** 2-3 evenings.

## 5. MLX with `mx.compile` &mdash; can the dispatch overhead be controlled?

**Goal.** Test whether MLX-GPU's loss to pure Python (1,865 tok/sec vs 4,332) is *fundamental* or
*configurational*. By default MLX is eager &mdash; one kernel launch per op &mdash; and there are ~15 ops per token,
so per-op launch latency × 15 sets the floor. `mx.compile` fuses the whole graph into one launch per token.

**Why.** The talos-vs-macbook benchmark left this question open. Two distinct outcomes are interesting:
- **If MLX-GPU jumps to 50-200K tok/sec with `mx.compile`:** the lesson sharpens to "framework dispatch
  overhead is always *optional* &mdash; you just have to ask for it." Makes the WASM win look less surprising.
- **If MLX-GPU barely moves:** the lesson sharpens the other way &mdash; "Apple Silicon GPU launch cost is
  per-token-fundamental, not per-op." Makes the FPGA's deterministic latency look more important.

Either result is publishable. The experiment is cheap (1-line change to `bench_mlx.py` plus warmup) and
takes one evening.

**Build path.**
1. Add a `bench_mlx_compiled.py` that wraps the forward pass in `@mx.compile` (and a JIT-warmup phase).
2. Compare numbers against the existing eager-MLX result; report both.
3. If the compiled version wins big, also try `mx.compile` with batched parallel streams.

**Estimated effort.** 1 evening.

## 6. Linux / Raspberry Pi 5 deployment (optional)

**Goal.** Run the existing `bench_c` unmodified on a Cortex-A76 with NEON.

**Why.** Closes the "what does NEON cost on a non-Apple ARM core?" question. Pi 5 is ~$80 in India. Useful as a development target for Om and as another data point on the chart.

**Build path.** Just `make && ./bench_c`. No code changes.

**Estimated effort.** 1 hour once Pi arrives.

## Order of attack

1. WebAssembly &mdash; **DONE, live**.
2. **MLX `mx.compile` experiment (1 evening)** &mdash; the next cheap, high-information experiment.
3. Microcontrollers when boards arrive (1 weekend).
4. CoreML / ANE (1 evening).
5. Metal fused-kernel (2-3 evenings).
6. Pi 5 (drop-in, whenever).
