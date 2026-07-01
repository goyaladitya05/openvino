// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

// Marks decomposed RMSNorm chains (mean(x^2) → eps → sqrt → rsqrt → normalize)
// so EnforceInferencePrecision keeps them in f32 under bf16/f16 inference.
//
// Required because RMSFusion(enable_without_gamma=false) cannot fuse affine-free
// RMSNorm (e.g. LTX-Video norm1/norm2), leaving the chain decomposed.  PyTorch's
// _fused_rms_norm always upcasts bf16 input to f32 internally; OV's decomposed
// path does not, causing variance collapse across many transformer blocks.
class KeepRMSNormPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::intel_cpu::KeepRMSNormPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
