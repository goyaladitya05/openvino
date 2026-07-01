// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "keep_rms_norm_precision.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "snippets/pass/tokenization.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

static std::shared_ptr<ov::Node> single_consumer(const ov::Output<ov::Node>& output) {
    const auto& targets = output.get_target_inputs();
    if (targets.size() != 1)
        return nullptr;
    return targets.begin()->get_node()->shared_from_this();
}

static std::optional<float> scalar_float(const std::shared_ptr<ov::Node>& node) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!c)
        return std::nullopt;
    if (ov::shape_size(c->get_shape()) != 1)
        return std::nullopt;
    return c->cast_vector<float>()[0];
}

// Returns true if node is a Constant whose every element equals `val`.
static bool is_all_value(const std::shared_ptr<ov::Node>& node, float val) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!c)
        return false;
    if (ov::shape_size(c->get_shape()) == 0)
        return false;
    auto vals = c->cast_vector<float>();
    return std::all_of(vals.begin(), vals.end(), [val](float v) {
        return v == val;
    });
}

static void mark_f32(const std::shared_ptr<ov::Node>& node) {
    if (!ov::is_conversion_disabled(node, ov::element::f16))
        ov::disable_conversion(node, ov::element::f16);
}

bool KeepRMSNormPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int matched = 0;

    for (const auto& node : model->get_ordered_ops()) {
        // Anchor: ReduceMean whose input is Power(x, 2.0f)
        auto reduce_mean = ov::as_type_ptr<ov::op::v1::ReduceMean>(node);
        if (!reduce_mean)
            continue;

        auto power_sq = ov::as_type_ptr<ov::op::v1::Power>(reduce_mean->get_input_node_shared_ptr(0));
        if (!power_sq)
            continue;

        auto exp_val = scalar_float(power_sq->get_input_node_shared_ptr(1));
        if (!exp_val || *exp_val != 2.0f)
            continue;

        // Add(mean, eps): single consumer of ReduceMean must be Add with a scalar const
        auto add_eps = ov::as_type_ptr<ov::op::v1::Add>(single_consumer(reduce_mean->output(0)));
        if (!add_eps)
            continue;
        bool has_eps_const = scalar_float(add_eps->get_input_node_shared_ptr(0)).has_value() ||
                             scalar_float(add_eps->get_input_node_shared_ptr(1)).has_value();
        if (!has_eps_const)
            continue;

        auto after_add = single_consumer(add_eps->output(0));
        if (!after_add)
            continue;

        std::shared_ptr<ov::Node> rsqrt_node;

        if (auto sqrt_node = ov::as_type_ptr<ov::op::v0::Sqrt>(after_add)) {
            auto sqrt_consumer = single_consumer(sqrt_node->output(0));
            if (!sqrt_consumer)
                continue;
            if (auto inv = ov::as_type_ptr<ov::op::v1::Power>(sqrt_consumer)) {
                // (a1) Standard: Sqrt → Power(sqrt, -1)
                auto inv_exp = scalar_float(inv->get_input_node_shared_ptr(1));
                if (!inv_exp || *inv_exp != -1.0f)
                    continue;
                mark_f32(sqrt_node);
                rsqrt_node = inv;
            } else if (auto div = ov::as_type_ptr<ov::op::v1::Divide>(sqrt_consumer)) {
                // (a2) aten::rsqrt → Divide(Constant(1), Sqrt): numerator must be all-ones
                // Note: use is_all_value instead of scalar_float to handle non-scalar broadcast constants.
                if (!is_all_value(div->get_input_node_shared_ptr(0), 1.0f))
                    continue;
                mark_f32(sqrt_node);
                rsqrt_node = div;
            } else {
                continue;
            }
        } else if (auto fused = ov::as_type_ptr<ov::op::v1::Power>(after_add)) {
            // (b) Fused: Power(add_eps, -0.5)
            auto fused_exp = scalar_float(fused->get_input_node_shared_ptr(1));
            if (!fused_exp || *fused_exp != -0.5f)
                continue;
            rsqrt_node = fused;
        } else {
            continue;
        }

        // Find the normalize Multiply, possibly separated from rsqrt_node by a
        // type-normalizing Convert(f32→bf16/f16) that OV's PyTorch FE inserts to
        // unify mixed-dtype operands (PyTorch promotes bf16*f32→f32; OV IR does not).
        auto rsqrt_consumer = single_consumer(rsqrt_node->output(0));
        if (!rsqrt_consumer)
            continue;

        std::shared_ptr<ov::op::v0::Convert> type_norm_convert;
        if (auto cv = ov::as_type_ptr<ov::op::v0::Convert>(rsqrt_consumer)) {
            auto out_type = cv->get_output_element_type(0);
            if (out_type == ov::element::bf16 || out_type == ov::element::f16) {
                type_norm_convert = cv;
                rsqrt_consumer = single_consumer(cv->output(0));
                if (!rsqrt_consumer)
                    continue;
            }
        }

        auto mul_norm = ov::as_type_ptr<ov::op::v1::Multiply>(rsqrt_consumer);
        if (!mul_norm)
            continue;

        // Determine which Multiply port carries the rsqrt chain vs. hidden_states.
        std::shared_ptr<ov::Node> rsqrt_chain_end =
            type_norm_convert ? std::static_pointer_cast<ov::Node>(type_norm_convert) : rsqrt_node;
        int rsqrt_port = -1;
        for (int p = 0; p < 2; ++p) {
            if (mul_norm->get_input_node_shared_ptr(p) == rsqrt_chain_end)
                rsqrt_port = p;
        }
        if (rsqrt_port == -1)
            continue;
        const int hidden_port = 1 - rsqrt_port;

        // Mark variance chain nodes f32 (variance, eps, sqrt, rsqrt stay f32).
        mark_f32(power_sq);
        mark_f32(reduce_mean);
        mark_f32(add_eps);
        mark_f32(after_add);
        mark_f32(rsqrt_node);

        // Force the normalize Multiply to compute in f32, matching PyTorch's type-promotion
        // semantics where bf16 * f32 → f32 (Diffusers RMSNorm relies on this).
        //
        // OV's PyTorch FE inserts Convert(f32→bf16) on the rsqrt path, producing
        // Multiply(bf16, bf16) → bf16 in the IR.  We fix this by:
        //   1. Bypassing that Convert so rsqrt_node's f32 output goes directly to the Multiply.
        //   2. Inserting Convert(bf16→f32) on the hidden_states input.
        // Result: Multiply(f32, f32) → f32, whose output is then Reorder'd to bf16 by
        // EnforceInferencePrecision for all downstream mandatory-bf16 ops.
        if (type_norm_convert) {
            mul_norm->input(rsqrt_port).replace_source_output(rsqrt_node->output(0));
            snippets::pass::SetSnippetsNodeType(type_norm_convert, snippets::pass::SnippetsNodeType::SkippedByPlugin);
        }
        auto hidden_input = mul_norm->input_value(hidden_port);
        if (hidden_input.get_element_type() != ov::element::f32) {
            auto upcast = std::make_shared<ov::op::v0::Convert>(hidden_input, ov::element::f32);
            upcast->validate_and_infer_types();
            mul_norm->input(hidden_port).replace_source_output(upcast->output(0));
        }
        mul_norm->validate_and_infer_types();  // output type is now f32
        mark_f32(mul_norm);
        // Prevent Snippets from fusing [Convert(bf16→f32), Multiply, Convert(f32→bf16)] into a
        // Subgraph that runs in bf16 externally, which would override our f32 precision marking.
        snippets::pass::SetSnippetsNodeType(mul_norm, snippets::pass::SnippetsNodeType::SkippedByPlugin);
        changed = true;
        ++matched;
    }

    std::cerr << "[KeepRMSNormPrecision] matched " << matched << " chains\n";
    return changed;
}

}  // namespace ov::intel_cpu
