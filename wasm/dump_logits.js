// Run the actual WASM module under Node and dump logits for fixed inputs.
// Output is consumed by verify_against_numpy.py for live correctness checking.
//
// Usage: node dump_logits.js > logits.json

const fs = require('node:fs');
const path = require('node:path');

const createMicroGPT = require('./microgpt_inf.js');

(async () => {
    const M = await createMicroGPT();

    const wbuf = fs.readFileSync(path.join(__dirname, 'weights.bin'));
    const ptr = M._get_weights_ptr();
    M.HEAPU8.set(new Uint8Array(wbuf.buffer, wbuf.byteOffset, wbuf.byteLength), ptr);
    M._init_weights();

    function logitsAt(tok, pos) {
        const lp = M._malloc(27 * 4);
        M._forward(tok, pos, lp);
        const out = Array.from(M.HEAPF32.subarray(lp >> 2, (lp >> 2) + 27));
        M._free(lp);
        return out;
    }

    // Single-step: BOS at position 0 with empty KV cache
    M._clear_kv();
    const first = logitsAt(26, 0);

    // Autoregressive trace: BOS, e, m, m, a (== "emma" with leading BOS)
    M._clear_kv();
    const seq = [26, 4, 12, 12, 0];
    const trace = seq.map((tok, pos) => ({ tok, pos, logits: logitsAt(tok, pos) }));

    // Speed: a tight benchmark window for a stability cross-check
    const tps_runs = [];
    M._benchmark(20000);  // warmup
    for (let i = 0; i < 5; i++) {
        const t = M._benchmark(100000);
        tps_runs.push(100000 / t);
    }

    process.stdout.write(JSON.stringify({
        first_logits: first,
        autoregressive_trace: trace,
        tps_runs,
    }, null, 2));
})();
