# LTX-Video CPU bf16 fix — server validation

Branch: `fix/cpu-bf16-rms-no-gamma`. The fix fuses affine-free RMSNorm
(`norm_elementwise_affine=False`, LTX norm1/norm2) into the internal RMS op so its JIT kernel
computes the variance reduction in f32 under bf16 enforcement, matching PyTorch's
`_fused_rms_norm` numerics (bf16 I/O, f32 inside, one rounding at the output).

## 1. Build

```bash
git fetch origin fix/cpu-bf16-rms-no-gamma && git checkout fix/cpu-bf16-rms-no-gamma
cmake --build build --target openvino_intel_cpu_plugin ov_cpu_func_tests --parallel
# use the in-tree python: export PYTHONPATH=<repo>/bin/intel64/Release/python
```

## 2. Unit/functional regression tests (AMX server runs the bf16 variants for real)

```bash
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter='*smoke_RMSNorm*'
```

Covers: gamma-less RMS single-layer (`smoke_RMSNorm_NoScale_CPU`), the LTX norm1/norm2
subgraph fusion + accuracy under bf16 (`smoke_RMSNormNoGamma_CPU`), plus the pre-existing
RMS suites (no regression).

## 3. Graph check on the real transformer IR (fast, no inference loop)

```bash
python validation_ltx_bf16/check_fusion_ltx.py \
    ~/run_i2v/ltx-video-ov-fp32-2d-ts/transformer/openvino_model.xml
```

Expect 168 RMS nodes (112 norm_q/k + 56 norm1/norm2) and no leftover decomposed
`norm1/norm2` Eltwise/Reduce ops.

## 4. End-to-end 30-step validation (the actual deliverable metric)

```bash
python validation_ltx_bf16/ltx_e2e_validate.py --model ~/run_i2v/ltx-video-ov-fp32-2d-ts
```

PASS = bf16 `std[last]/std[first]` within 5% of the f32 ratio (broken baseline was 0.79 vs
healthy 0.95+) and final rel-L2 far below the broken ~0.84, with bf16 still faster than f32.

Synthetic single-norm sanity check (any bf16 machine): `python bf16_rms_norm_diag.py --numeric`
(script at repo root, not committed).

## If e2e still shows a gap

The remaining suspects from the per-op audit, in priority order:
1. `norm_out` affine-free LayerNorm → runs as a single `MVN` node in bf16 (check its kernel's
   internal accumulation, or keep it f32 — it executes once per step).
2. RoPE table computation (`__module.rope/aten::div|mul` run bf16; positional angles rounded
   to bf16 before sin/cos would scramble high-frequency components).
Use `bf16_precision_search.py compare` (repo root) with OV_CPU_BLOB_DUMP on block 0 to rank
the first diverging layer.
