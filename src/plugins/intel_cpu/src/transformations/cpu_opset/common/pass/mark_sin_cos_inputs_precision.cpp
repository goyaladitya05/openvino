// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mark_sin_cos_inputs_precision.hpp"

#include <deque>
#include <memory>
#include <unordered_set>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "snippets/pass/tokenization.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

bool MarkSinCosInputsPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MarkSinCosInputsPrecision);

    auto is_traversal_boundary = [](const std::shared_ptr<ov::Node>& node) {
        // constants/parameters do not compute anything; compute-heavy ops must stay in low precision
        return ov::is_type_any_of<ov::op::v0::Constant,
                                  ov::op::v0::Parameter,
                                  ov::op::v0::MatMul,
                                  ov::op::v1::Convolution,
                                  ov::op::v1::GroupConvolution,
                                  ov::op::v1::ConvolutionBackpropData,
                                  ov::op::v1::GroupConvolutionBackpropData>(node);
    };

    auto mark = [](const std::shared_ptr<ov::Node>& node) {
        ov::disable_conversion(node, ov::element::f16);
        // keep the node out of Snippets Subgraphs: tokenized bodies are precision-enforced as a whole
        snippets::pass::SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
    };

    std::unordered_set<ov::Node*> visited;
    std::deque<std::shared_ptr<ov::Node>> to_visit;
    bool changed = false;

    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type_any_of<ov::op::v0::Sin, ov::op::v0::Cos>(op) || !op->get_output_element_type(0).is_real()) {
            continue;
        }
        if (visited.insert(op.get()).second) {
            mark(op);
            changed = true;
            to_visit.push_back(op);
        }
    }

    while (!to_visit.empty()) {
        const auto node = to_visit.front();
        to_visit.pop_front();

        for (const auto& input : node->input_values()) {
            const auto& producer = input.get_node_shared_ptr();
            if (!visited.insert(producer.get()).second) {
                continue;
            }
            if (is_traversal_boundary(producer)) {
                continue;
            }
            // integer/bool subpaths (shapes, indices) are not affected by low-precision enforcement
            if (!producer->get_output_element_type(0).is_real()) {
                continue;
            }
            mark(producer);
            to_visit.push_back(producer);
        }
    }

    return changed;
}

}  // namespace ov::intel_cpu
