"""End-to-end LTX-Video bf16-vs-f32 validation on the OpenVINO CPU plugin.

Runs the same 30-step generation twice (f32 reference, then bf16), records the latent
trajectory at every denoise step via callback, and reports:
  - per-step latent std for both precisions
  - per-step rel-L2 distance of the bf16 latents from the f32 latents
  - summary: std[last]/std[first] ratio (the variance-collapse metric) and wall time

Pass criteria (from the investigation baseline):
  - bf16 std ratio within ~5% of the f32 std ratio (f32 is ~0.95-1.0; broken bf16 was ~0.79)
  - final-step rel-L2 well below the broken baseline (~0.84)

Usage:
    python ltx_e2e_validate.py --model /path/to/ltx-video-ov-fp32-2d-ts \
        [--steps 30] [--height 320] [--width 480] [--frames 33] [--seed 42] [--save-videos]
"""
import argparse
import time

import numpy as np
import torch


def load_pipeline(model_dir: str, precision: str):
    from optimum.intel import OVLTXPipeline

    return OVLTXPipeline.from_pretrained(
        model_dir,
        device="CPU",
        ov_config={"INFERENCE_PRECISION_HINT": precision},
    )


def run_once(model_dir: str, precision: str, args) -> dict:
    pipe = load_pipeline(model_dir, precision)

    latents_per_step = []

    def cb(pipe_, i, t, kwargs):
        lat = kwargs["latents"]
        latents_per_step.append(lat.detach().to(torch.float32).cpu().numpy().copy())
        return {}

    gen = torch.Generator().manual_seed(args.seed)
    call_kwargs = dict(
        prompt="a red fox running through a snowy forest, cinematic",
        negative_prompt="worst quality, blurry, jittery, distorted",
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        generator=gen,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="np",
    )

    t0 = time.perf_counter()
    result = pipe(**call_kwargs)
    dt = time.perf_counter() - t0

    frames = np.asarray(result.frames[0])
    if args.save_videos:
        np.save(f"video_{precision}.npy", frames)
    del pipe
    return {
        "latents": latents_per_step,
        "stds": [float(a.std()) for a in latents_per_step],
        "time_s": dt,
        "frames_std": float(frames.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-videos", action="store_true")
    args = ap.parse_args()

    import openvino as ov

    print("OV:", ov.__version__)
    caps = ov.Core().get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    if "BF16" not in caps:
        raise SystemExit("This CPU has no native BF16 — run on the bf16-capable server.")

    print("\n--- f32 reference run ---")
    ref = run_once(args.model, "f32", args)
    print(f"f32:  {ref['time_s']:.1f}s, frames std {ref['frames_std']:.4f}")

    print("\n--- bf16 run ---")
    test = run_once(args.model, "bf16", args)
    print(f"bf16: {test['time_s']:.1f}s, frames std {test['frames_std']:.4f}")

    n = min(len(ref["latents"]), len(test["latents"]))
    print(f"\n{'step':>4}  {'std f32':>9}  {'std bf16':>9}  {'rel-L2 bf16 vs f32':>19}")
    rel_l2 = []
    for i in range(n):
        a, b = ref["latents"][i], test["latents"][i]
        r = float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))
        rel_l2.append(r)
        print(f"{i:>4}  {ref['stds'][i]:>9.4f}  {test['stds'][i]:>9.4f}  {r:>19.4f}")

    ratio_ref = ref["stds"][-1] / ref["stds"][0]
    ratio_test = test["stds"][-1] / test["stds"][0]
    print("\n=== summary ===")
    print(f"f32  std[last]/std[first] = {ratio_ref:.4f}")
    print(f"bf16 std[last]/std[first] = {ratio_test:.4f}   (broken baseline ~0.79 vs healthy ~0.95+)")
    print(f"final rel-L2 = {rel_l2[-1]:.4f}                (broken baseline ~0.84)")
    print(f"speed: f32 {ref['time_s']:.1f}s -> bf16 {test['time_s']:.1f}s "
          f"({ref['time_s'] / max(test['time_s'], 1e-9):.2f}x)")
    passed = ratio_test >= 0.95 * ratio_ref and rel_l2[-1] < 0.25
    print("RESULT:", "PASS" if passed else "FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
