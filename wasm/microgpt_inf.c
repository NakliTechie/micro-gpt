// microGPT inference in WebAssembly.
// Same forward pass as benchmark/talos-vs-macbook-m5-pro/bench_numpy.py:
//   tok+pos -> rmsnorm -> rmsnorm -> Q/K/V -> 4-head attn -> WO + residual
//   -> rmsnorm -> ReLU MLP (4x) + residual -> lm_head -> logits
// 4,192 fp32 weights loaded once at init from a binary blob.
//
// Build:  emcc -O3 -msimd128 -ffast-math microgpt_inf.c -o microgpt_inf.js \
//          -sEXPORTED_FUNCTIONS=_get_weights_ptr,_init_weights,_forward,_clear_kv,_generate_name \
//          -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPF32,HEAPU8 \
//          -sMODULARIZE=1 -sEXPORT_NAME=createMicroGPT -sALLOW_MEMORY_GROWTH=1

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <emscripten/emscripten.h>

#define VOCAB_SIZE 27
#define BLOCK_SIZE 16
#define N_HEAD     4
#define N_EMBD     16
#define HEAD_DIM   4               // N_EMBD / N_HEAD
#define MLP_HIDDEN 64              // 4 * N_EMBD
#define BOS        26
#define TOTAL_PARAMS 4192

static float weights_buf[TOTAL_PARAMS];
static const float *WTE, *WPE;
static const float *WQ, *WK, *WV, *WO;
static const float *W1, *W2;
static const float *LM;

static float K_cache[BLOCK_SIZE * N_EMBD];
static float V_cache[BLOCK_SIZE * N_EMBD];

EMSCRIPTEN_KEEPALIVE
float* get_weights_ptr(void) { return weights_buf; }

EMSCRIPTEN_KEEPALIVE
void init_weights(void) {
    WTE = weights_buf;
    WPE = WTE + 27 * 16;
    WQ  = WPE + 16 * 16;
    WK  = WQ  + 16 * 16;
    WV  = WK  + 16 * 16;
    WO  = WV  + 16 * 16;
    W1  = WO  + 16 * 16;
    W2  = W1  + 64 * 16;
    LM  = W2  + 16 * 64;
}

EMSCRIPTEN_KEEPALIVE
void clear_kv(void) {
    memset(K_cache, 0, sizeof(K_cache));
    memset(V_cache, 0, sizeof(V_cache));
}

static inline void rmsnorm(float *x) {
    // Reference (bench_numpy.py): scale = 1 / sqrt(mean(x*x) + eps)
    float ss = 0.0f;
    for (int i = 0; i < N_EMBD; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)N_EMBD + 1e-5f);
    for (int i = 0; i < N_EMBD; i++) x[i] *= scale;
}

// y = W @ x, W is (out, in) row-major.
static inline void matvec(float *y, const float *W, const float *x, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float s = 0.0f;
        const float *row = W + i * in_dim;
        for (int j = 0; j < in_dim; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

// Single-token forward. Updates K[pos], V[pos]. Writes 27-dim logits.
EMSCRIPTEN_KEEPALIVE
void forward(int tok, int pos, float *logits_out) {
    float x[N_EMBD], xr[N_EMBD];
    float q[N_EMBD], kk[N_EMBD], vv[N_EMBD];
    float head_out[N_EMBD];
    float h[MLP_HIDDEN];
    float wo_out[N_EMBD], w2_out[N_EMBD];

    for (int i = 0; i < N_EMBD; i++) {
        x[i] = WTE[tok * N_EMBD + i] + WPE[pos * N_EMBD + i];
    }
    rmsnorm(x);

    memcpy(xr, x, sizeof(x));
    rmsnorm(x);  // pre-attention norm (reference does the same redundant 2nd norm)

    matvec(q,  WQ, x, N_EMBD, N_EMBD);
    matvec(kk, WK, x, N_EMBD, N_EMBD);
    matvec(vv, WV, x, N_EMBD, N_EMBD);
    memcpy(&K_cache[pos * N_EMBD], kk, sizeof(kk));
    memcpy(&V_cache[pos * N_EMBD], vv, sizeof(vv));

    const float inv_sqrt_hd = 1.0f / sqrtf((float)HEAD_DIM);
    const int seq_len = pos + 1;
    for (int hd = 0; hd < N_HEAD; hd++) {
        float scores[BLOCK_SIZE];
        float maxs = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            float s = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                s += q[hd * HEAD_DIM + d] * K_cache[t * N_EMBD + hd * HEAD_DIM + d];
            }
            s *= inv_sqrt_hd;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - maxs);
            sum += scores[t];
        }
        float inv_sum = 1.0f / sum;
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                acc += scores[t] * inv_sum * V_cache[t * N_EMBD + hd * HEAD_DIM + d];
            }
            head_out[hd * HEAD_DIM + d] = acc;
        }
    }

    matvec(wo_out, WO, head_out, N_EMBD, N_EMBD);
    for (int i = 0; i < N_EMBD; i++) x[i] = wo_out[i] + xr[i];

    memcpy(xr, x, sizeof(x));
    rmsnorm(x);
    matvec(h, W1, x, MLP_HIDDEN, N_EMBD);
    for (int i = 0; i < MLP_HIDDEN; i++) if (h[i] < 0.0f) h[i] = 0.0f;
    matvec(w2_out, W2, h, N_EMBD, MLP_HIDDEN);
    for (int i = 0; i < N_EMBD; i++) x[i] = w2_out[i] + xr[i];

    matvec(logits_out, LM, x, VOCAB_SIZE, N_EMBD);
}

// xorshift PRNG (matches the C benchmark for cross-check)
static uint64_t rng_state = 42;
static inline uint64_t xorshift(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double uniform(void) {
    return (xorshift() >> 11) * (1.0 / 9007199254740992.0);
}

EMSCRIPTEN_KEEPALIVE
void seed_rng(uint64_t s) { rng_state = s ? s : 1; }

// Generate one full name (until BOS or max_len). Writes character ids to
// out_tokens, returns length. Temperature-0.5 multinomial sampling.
EMSCRIPTEN_KEEPALIVE
int generate_name(int max_len, int *out_tokens) {
    clear_kv();
    float logits[VOCAB_SIZE];
    int tok = BOS;
    int pos = 0;
    int n = 0;

    while (n < max_len && pos < BLOCK_SIZE) {
        forward(tok, pos, logits);

        // Softmax with temp=0.5 -> divide logits by 0.5 i.e. multiply by 2
        float maxs = -1e30f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] *= 2.0f;
            if (logits[i] > maxs) maxs = logits[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] = expf(logits[i] - maxs);
            sum += logits[i];
        }
        double r = uniform() * (double)sum;
        double cum = 0.0;
        int next = VOCAB_SIZE - 1;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            cum += logits[i];
            if (r < cum) { next = i; break; }
        }

        if (next == BOS) break;
        out_tokens[n++] = next;
        tok = next;
        pos++;
    }
    return n;
}

// Benchmark: run n_tokens single-token forwards (resetting every time we
// hit BOS or the block boundary) and return total elapsed time in seconds.
EMSCRIPTEN_KEEPALIVE
double benchmark(int n_tokens) {
    clear_kv();
    float logits[VOCAB_SIZE];
    int tok = BOS;
    int pos = 0;

    double t0 = emscripten_get_now();
    for (int step = 0; step < n_tokens; step++) {
        if (pos >= BLOCK_SIZE) {
            clear_kv();
            tok = BOS;
            pos = 0;
        }
        forward(tok, pos, logits);

        float maxs = -1e30f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] *= 2.0f;
            if (logits[i] > maxs) maxs = logits[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] = expf(logits[i] - maxs);
            sum += logits[i];
        }
        double r = uniform() * (double)sum;
        double cum = 0.0;
        int next = VOCAB_SIZE - 1;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            cum += logits[i];
            if (r < cum) { next = i; break; }
        }
        if (next == BOS) {
            clear_kv();
            tok = BOS;
            pos = 0;
        } else {
            tok = next;
            pos++;
        }
    }
    double t1 = emscripten_get_now();
    return (t1 - t0) / 1000.0;  // ms -> s
}
