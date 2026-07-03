"""Graph-level check on the real LTX transformer IR: the 56 affine-free norm1/norm2 chains
must fuse into RMS nodes (in addition to the 112 norm_q/norm_k that already fused).

Usage:
    python check_fusion_ltx.py /path/to/ltx-video-ov/transformer/openvino_model.xml
"""
import sys
from collections import Counter

import openvino as ov


def run(model_xml: str, precision: str) -> None:
    core = ov.Core()
    model = core.read_model(model_xml)
    compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": precision})

    rms = Counter()
    leftovers = []
    mvn = []
    for node in compiled.get_runtime_model().get_ops():
        rt = node.get_rt_info()
        lt = rt["layerType"].astype(str)
        rp = rt["runtimePrecision"].astype(str)
        orig = rt["originalLayersNames"].astype(str) if "originalLayersNames" in rt else ""
        if lt == "RMS":
            rms[rp] += 1
        if lt in ("Eltwise", "Reduce", "Subgraph") and ("norm1/aten::pow" in orig or "norm2/aten::pow" in orig
                                                        or "norm1/aten::mean" in orig or "norm2/aten::mean" in orig):
            leftovers.append((lt, rp, orig[:90]))
        if lt == "MVN":
            mvn.append((rp, orig[:90]))

    print(f"\n=== {precision} ===")
    print(f"RMS nodes by runtime precision: {dict(rms)}  (expect 168 total: 112 norm_q/k + 56 norm1/norm2)")
    if leftovers:
        print(f"REMAINING decomposed norm1/norm2 ops ({len(leftovers)}) — fusion did NOT fire for these:")
        for lt, rp, o in leftovers[:20]:
            print(f"  [{lt} {rp}] {o}")
    else:
        print("OK: no decomposed norm1/norm2 ops remain in the exec graph")
    for rp, o in mvn:
        print(f"note: MVN (norm_out) runs in {rp}: {o}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print("OV:", ov.__version__)
    run(sys.argv[1], "f32")
    caps = ov.Core().get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    if "BF16" in caps:
        run(sys.argv[1], "bf16")
    else:
        print("\n(bf16 check skipped — no native BF16 on this CPU)")
