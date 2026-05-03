# Pending experiments

Replication targets for the same 4,192-parameter microGPT model across additional substrates.
Each one closes a specific data point in the "where does time actually go" map.

## 1. WebAssembly  &mdash; DONE

**Result.** **1,335,113 tok/sec** in Chrome on the M4 Pro &mdash; **25.19&times; the FPGA**, **35% of native C+NEON** (3.82M).

**What this told us.** The "runtime vs compiled" cliff hypothesis was correct: WASM (compiled, JIT-optimized) sits firmly on
the *fast* side of the cliff, three orders of magnitude above NumPy and MLX. The remaining 65% gap to native C is from
wasm-simd128 not perfectly mapping to NEON, plus browser sandbox overhead. Names generated match the native quality
("sona", "kana", "kashi", "dyan").

**Files.** `wasm/microgpt_inf.c`, `wasm/index.html`, `wasm/build.sh`. Open `wasm/index.html` from any modern browser
(after `python3 -m http.server` from the wasm dir). The artifact is portable &mdash; can host on GitHub Pages.

**Effort.** ~2 hours including emscripten install troubleshooting.

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

## 5. Linux / Raspberry Pi 5 deployment (optional)

**Goal.** Run the existing `bench_c` unmodified on a Cortex-A76 with NEON.

**Why.** Closes the "what does NEON cost on a non-Apple ARM core?" question. Pi 5 is ~$80 in India. Useful as a development target for Om and as another data point on the chart.

**Build path.** Just `make && ./bench_c`. No code changes.

**Estimated effort.** 1 hour once Pi arrives.

## Order of attack

1. WebAssembly (1 evening)  ← starting now
2. Microcontrollers when boards arrive (1 weekend)
3. CoreML / ANE (1 evening)
4. Metal fused-kernel (2-3 evenings)
5. Pi 5 (drop-in, whenever)
