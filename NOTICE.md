# NOTICE — third-party assets

The WASM demo bundles a copy of the trained model weights to make
GitHub Pages hosting self-contained. Provenance:

## `wasm/weights.bin`

A 16,768-byte flat fp32 dump of the microGPT weights originally trained
by **Luthira Abeykoon** as part of the
[TALOS-V2 project](https://github.com/Luthiraa/TALOS-V2). The trained
weights live upstream at `rtl/microgpt/weights_only.npy`. The flat binary
is produced by Alex Cheema's `convert_weights.py` in
[`talos-vs-macbook-m5-pro`](https://github.com/itsrealranky/talos-vs-macbook-m5-pro/blob/main/convert_weights.py).

The TALOS-V2 repository does not publish an explicit license. We
include `weights.bin` here on the same footing as the upstream public
hosting: derivative use for non-commercial educational and benchmarking
purposes, with attribution. If Luthira asks for the file to be removed
or replaced with a runtime-fetch from the upstream URL, that will be
done immediately.

## `wasm/microgpt_inf.js`, `wasm/microgpt_inf.wasm`

These are build artifacts produced by Emscripten from `wasm/microgpt_inf.c`
in this repository. They are committed so the GitHub Pages demo works
without a local build step. To regenerate after editing
`microgpt_inf.c`:

```sh
cd wasm && ./build.sh
```

## CDN dependency

`report/index.html` loads Plotly.js from
`https://cdn.plot.ly/plotly-2.35.2.min.js` at page load. No other
network dependencies; the WASM demo runs entirely offline once loaded.
