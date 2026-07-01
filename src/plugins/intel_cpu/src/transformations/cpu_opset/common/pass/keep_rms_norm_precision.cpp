// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "keep_rms_norm_precision.hpp"

#include <memory>
#include <optional>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

// Returns the single consumer of `output`, or nullptr if there are zero or more than one.
static std::shared_ptr<ov::Node> single_consumer(const ov::Output<ov::Node>& output) {
    const auto& targets = output.get_target_inputs();
    if (targets.size() != 1)
        return nullptr;
    return targets.begin()->get_node()->shared_from_this();
}

// Returns the scalar float value of a Constant node, or nullopt.
static std::optional<float> scalar_float(const std::shared_ptr<ov::Node>& node) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!c)
        return std::nullopt;
    if (ov::shape_size(c->get_shape()) != 1)
        return std::nullopt;
    return c->cast_vector<float>()[0];
}

static void mark_f32(const std::shared_ptr<ov::Node>& node) {
    if (!ov::is_conversion_disabled(node, ov::element::f16))
        ov::disable_conversion(node, ov::element::f16);
}

bool KeepRMSNormPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

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

        // ------- confirmed: mean(x^2) pattern -------

        // Add(mean, eps): the only consumer of ReduceMean must be Add with a scalar const
        auto add_eps = ov::as_type_ptr<ov::op::v1::Add>(single_consumer(reduce_mean->output(0)));
        if (!add_eps)
            continue;

        bool has_eps_const = scalar_float(add_eps->get_input_node_shared_ptr(0)).has_value() ||
                             scalar_float(add_eps->get_input_node_shared_ptr(1)).has_value();
        if (!has_eps_const)
            continue;

        // Next consumer of Add can be either:
        //   (a) Sqrt → Power(sqrt, -1)       — standard two-step rsqrt
        //   (b) Power(add_eps, -0.5)          — fused sqrt+rsqrt shortcut
        auto after_add = single_consumer(add_eps->output(0));
        if (!after_add)
            continue;

        std::shared_ptr<ov::Node> rsqrt_node;

        if (auto sqrt_node = ov::as_type_ptr<ov::op::v0::Sqrt>(after_add)) {
            // (a) Standard: Sqrt → Power(sqrt, -1)
            auto inv = ov::as_type_ptr<ov::op::v1::Power>(single_consumer(sqrt_node->output(0)));
            if (!inv)
                continue;
            auto inv_exp = scalar_float(inv->get_input_node_shared_ptr(1));
            if (!inv_exp || *inv_exp != -1.0f)
                continue;
            mark_f32(sqrt_node);
            rsqrt_node = inv;
        } else if (auto fused = ov::as_type_ptr<ov::op::v1::Power>(after_add)) {
            // (b) Fused: Power(add_eps, -0.5)
            auto fused_exp = scalar_float(fused->get_input_node_shared_ptr(1));
            if (!fused_exp || *fused_exp != -0.5f)
                continue;
            rsqrt_node = fused;
        } else {
            continue;
        }

        // Normalize: the single consumer of rsqrt_node must be Multiply(x, rsqrt) or Multiply(rsqrt, x)
        auto mul_norm = ov::as_type_ptr<ov::op::v1::Multiply>(single_consumer(rsqrt_node->output(0)));
        if (!mul_norm)
            continue;

        // All nodes identified — mark the entire chain f32.
        mark_f32(power_sq);
        mark_f32(reduce_mean);
        mark_f32(add_eps);
        mark_f32(after_add);  // Sqrt or fused Power(-0.5)
        mark_f32(rsqrt_node);
        mark_f32(mul_norm);
        changed = true;
    }

    return changed;
}

}  // namespace ov::intel_cpu
