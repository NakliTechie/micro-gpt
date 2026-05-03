# microGPT benchmark on Apple M4 Pro

Hardware: Apple M4 Pro, 14 cores (10 P + 4 E), macOS 15.6.1, clang 17.
Model: TALOS-V2 trained microGPT, 4,192 params, n_embd=16, n_head=4, block_size=16.

## Single-stream throughput

| Implementation | tok/sec | vs FPGA |
| --- | ---: | ---: |
| pure-python | 4,332 | 0.08x |
| numpy fp32 | 24,223 | 0.46x |
| mlx fp32 (cpu) | 3,873 | 0.07x |
| mlx fp32 (gpu) | 1,865 | 0.04x |
| TALOS-V2 (FPGA, 56 MHz) | 53,000 | 1.00x |
| c fp32+NEON | 3,820,760 | **72.09x** |
| c Q4.12 fixed-point | 2,191,219 | 41.34x |

## Multi-stream aggregate (NEON)

| Streams | tok/sec |
| ---: | ---: |
| 8 | 24,962,058 |
| 10 (P-cores) | 28,028,917 |
| 14 (all cores) | 32,894,149 |
| 18 | 32,647,139 |

Saturates around the 14-core mark; 18 doesn't add anything.

## Reference (M5 Pro published numbers)

For comparison: published M5 Pro benchmark fork reports
~6.7M single-stream and ~86M aggregate. M4 Pro is roughly 55-65% of M5 Pro
on these workloads.

## Profiling: where does time go?

### Pure Python (4.3K tok/sec)

cProfile `tottime` ranking:
- ~80% in `linear()` -- specifically the list comprehension
  doing `sum(a*b for a,b in zip(...))`
- 115,170 `sum()` calls, 2M generator iterations
- ~50 ns of Python overhead per multiply-accumulate

The overhead **is** the cost. Real math is ~10% of wall-clock.

### NumPy fp32 (24K tok/sec)

- Only ~4% in the actual matmul kernel (`c_einsum`)
- ~12% in `rmsnorm` Python wrapper
- ~12% in `_mean`, ~10% in `ufunc.reduce`
- The rest: dispatchers, `_count_reduce_items`, type checks, isinstance,
  scalar conversions

Per-op overhead ~5-30 us. With ~10 ops per token, that's 50-300 us per token,
matching the observed 24K tok/sec. **Framework dispatch dominates.**

### MLX (3.8K cpu, 1.9K gpu)

Worse than pure Python on this model. Same dispatch story but the GPU
launch overhead is wrong-shape for 4K MACs per token. GPUs are idle 99%
of the time waiting for the kernel to be queued.

### C + NEON (3.8M single, 33M aggregate)

The whole model + KV cache fits in L1. No framework, no dispatchers,
no type machinery. NEON intrinsics for the inner loops. The thing it
does that nothing else does: it gets out of its own way.

## Lesson

For a 4K-parameter model, the question is not "how fast can your hardware
multiply?" but "how much overhead does your stack add per multiply?"

- Python: 50 ns per MAC (Python overhead dominates the math 10:1)
- NumPy: 1-3 us per op (dispatch dominates the math 25:1)
- MLX/GPU: ~ms per launch (launch dominates the math 100,000:1)
- C/NEON: bare-metal, hits actual silicon throughput
- FPGA: same math, different silicon, more deterministic latency
