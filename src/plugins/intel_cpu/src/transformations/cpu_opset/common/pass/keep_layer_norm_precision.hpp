// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

// Marks decomposed affine LayerNorm chains (MVN → Multiply → Add) f32 under
// bf16/f16 inference, matching PyTorch's behaviour where F.layer_norm always
// computes the gamma/beta application in opmath_t (= float for bfloat16).
//
// OV's PyTorch FE decomposes aten::layer_norm into:
//   MVN (internally f32 via JIT kernel, but outputs bf16 when EIP forces bf16)
//   Multiply (gamma scaling, runs in bf16)
//   Add      (beta bias,  runs in bf16)
//
// Marking all three with disable_conversion(f16) prevents EIP from forcing
// them to bf16, so IOAPD selects f32 PDs and inserts Reorder(f32→bf16) only
// at the boundary with downstream bf16 ops (e.g. convolutions).
class KeepLayerNormPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::intel_cpu::KeepLayerNormPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
