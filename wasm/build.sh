#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Copy weights from the benchmark dir
cp ../benchmark/talos-vs-macbook-m5-pro/assets/weights_fp32.bin ./weights.bin

emcc -O3 -msimd128 -ffast-math microgpt_inf.c -o microgpt_inf.js \
    -sEXPORTED_FUNCTIONS='["_get_weights_ptr","_init_weights","_clear_kv","_forward","_generate_name","_benchmark","_seed_rng","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPU8","HEAP32"]' \
    -sMODULARIZE=1 -sEXPORT_NAME=createMicroGPT \
    -sALLOW_MEMORY_GROWTH=1 \
    -sENVIRONMENT=web

ls -lh microgpt_inf.{js,wasm} weights.bin
echo "build ok"
