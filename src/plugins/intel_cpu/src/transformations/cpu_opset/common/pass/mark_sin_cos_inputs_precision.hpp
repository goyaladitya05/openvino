// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Keeps the argument computation of Sin/Cos in f32 under bf16/f16 inference precision.
 *
 * Trigonometric functions are precision-sensitive to *argument* rounding: rotary position
 * embeddings (RoPE) compute angles up to theta*pi/2 (~1.6e4 rad for theta=1e4). bf16 has ~2^-8
 * relative precision, so an angle of that magnitude is quantized in steps of ~60 rad and
 * sin/cos of it is meaningless, which scrambles the high-frequency positional components
 * (PyTorch always computes such tables in f32). MarkFloatingPointRange only propagates its
 * protection through Convert/compare/reshape-like ops and loses the path at the first
 * arithmetic op, so runtime-computed angle chains (e.g. LTX-Video's 3-axis RoPE, where the
 * grid depends on width/height/num_frames inputs) still get lowered to bf16.
 *
 * This pass walks backward from every Sin/Cos node over its floating-point producer cone,
 * stopping at Constants, Parameters and compute-heavy ops (MatMul/Convolution families), and
 * marks the visited nodes with disable_conversion(f16) so both the bf16 node-level enforcement
 * and the f16 ConvertPrecision keep them in f32. Marked nodes are also excluded from Snippets
 * tokenization, which would otherwise hide them inside a Subgraph that gets enforced as a
 * whole. The cone consists of cheap once-per-inference elementwise/movement ops, so the
 * performance impact is negligible; MatMuls are never crossed and stay in low precision.
 */
class MarkSinCosInputsPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MarkSinCosInputsPrecision");
    MarkSinCosInputsPrecision() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
