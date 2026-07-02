// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "keep_layer_norm_precision.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "snippets/pass/tokenization.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

static std::shared_ptr<ov::Node> single_consumer_ln(const ov::Output<ov::Node>& output) {
    const auto& targets = output.get_target_inputs();
    if (targets.size() != 1)
        return nullptr;
    return targets.begin()->get_node()->shared_from_this();
}

static void mark_f32_ln(const std::shared_ptr<ov::Node>& node) {
    if (!ov::is_conversion_disabled(node, ov::element::f16))
        ov::disable_conversion(node, ov::element::f16);
}

bool KeepLayerNormPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int matched = 0;

    for (const auto& node : model->get_ordered_ops()) {
        // Anchor: MVN node from aten::layer_norm decomposition (v0 or v6).
        const bool is_mvn = ov::as_type_ptr<ov::op::v0::MVN>(node) ||
                            ov::as_type_ptr<ov::op::v6::MVN>(node);
        if (!is_mvn)
            continue;

        // MVN must feed a single Multiply (gamma scaling: output * gamma).
        auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(single_consumer_ln(node->output(0)));
        if (!mul)
            continue;

        // Mark MVN and Multiply f32 so EIP does not force them to bf16.
        mark_f32_ln(node);
        mark_f32_ln(mul);
        // Prevent Snippets from fusing the affine pair into a bf16 Subgraph.
        snippets::pass::SetSnippetsNodeType(mul, snippets::pass::SnippetsNodeType::SkippedByPlugin);

        // Also mark the downstream Add (beta bias: * gamma + beta) if present.
        if (auto add = ov::as_type_ptr<ov::op::v1::Add>(single_consumer_ln(mul->output(0)))) {
            mark_f32_ln(add);
            snippets::pass::SetSnippetsNodeType(add, snippets::pass::SnippetsNodeType::SkippedByPlugin);
        }

        changed = true;
        ++matched;
    }

    std::cerr << "[KeepLayerNormPrecision] matched " << matched << " chains\n";
    return changed;
}

}  // namespace ov::intel_cpu
