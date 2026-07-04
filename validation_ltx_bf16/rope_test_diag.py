"""Replicates the RopeTablePrecision gtest model and diagnoses the bf16 error.

Prints (a) exec-graph layer types + runtime precisions for the angle chain, (b) the error
distribution of bf16 vs f32-CPU and vs f64 numpy, so we can tell a calibration problem
(smooth tail) from a marking gap (angle chain in bf16 / structured outliers).

Usage: python validation_ltx_bf16/rope_test_diag.py
"""
import numpy as np
import openvino as ov
import openvino.opset10 as ops

N, H = 128, 64
rng = np.random.default_rng(7)
W1 = (rng.random((H, H), dtype=np.float32) * 2 - 1)
W2 = (rng.random((H, H), dtype=np.float32) * 2 - 1)
freqs = (0.01 * np.power(400.0, np.arange(H, dtype=np.float32) / H)).reshape(1, H).astype(np.float32)


def build():
    pos = ops.parameter(ov.PartialShape([-1, 1]), ov.Type.f32, name="positions")
    x = ops.parameter(ov.PartialShape([-1, H]), ov.Type.f32, name="x")
    angles = ops.multiply(pos, ops.constant(freqs))
    cos, sin = ops.cos(angles), ops.sin(angles)
    q = ops.matmul(x, ops.constant(W1), False, False)
    rotated = ops.add(ops.multiply(q, cos), ops.multiply(q, sin))
    out = ops.matmul(rotated, ops.constant(W2), False, False)
    return ov.Model([out], [pos, x], "RopeDiag")


core = ov.Core()
pos_v = np.arange(N, dtype=np.float32).reshape(N, 1)
x_v = (rng.random((N, H), dtype=np.float32) * 2 - 1)

print("OV:", ov.__version__)
res = {}
for prec in ("f32", "bf16"):
    compiled = core.compile_model(build(), "CPU", {"INFERENCE_PRECISION_HINT": prec})
    res[prec] = compiled((pos_v, x_v))[0].copy()
    print(f"\n=== exec graph ({prec}) ===")
    for node in compiled.get_runtime_model().get_ordered_ops():
        rt = node.get_rt_info()
        lt = rt["layerType"].astype(str)
        rp = rt["runtimePrecision"].astype(str)
        if lt in ("Input", "Output", "Const"):
            continue
        orig = rt["originalLayersNames"].astype(str) if "originalLayersNames" in rt else ""
        print(f"  {lt:16s} {rp:6s} {node.get_friendly_name()[:44]:46s} <- {orig[:44]}")

# f64 numpy reference
angles64 = pos_v.astype(np.float64) @ freqs.astype(np.float64).reshape(1, H)
q64 = x_v.astype(np.float64) @ W1.astype(np.float64)
rot64 = q64 * np.cos(angles64) + q64 * np.sin(angles64)
ref64 = rot64 @ W2.astype(np.float64)

for name, a, b in (("bf16 vs f32-CPU", res["bf16"], res["f32"]),
                   ("bf16 vs f64-numpy", res["bf16"], ref64),
                   ("f32-CPU vs f64-numpy", res["f32"], ref64)):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    rel = d / (np.abs(b) + 1e-6)
    thr = 0.05 + 0.05 * np.abs(b)  # the gtest comparator
    print(f"\n--- {name} ---")
    print(f"  max abs {d.max():.4f} at {np.unravel_index(d.argmax(), d.shape)}, "
          f"max rel {rel.max():.4f}, mean abs {d.mean():.5f}")
    print(f"  elements over gtest threshold: {(d > thr).sum()} / {d.size}")
    for q_ in (50, 99, 99.9):
        print(f"  abs p{q_}: {np.percentile(d, q_):.4f}")
