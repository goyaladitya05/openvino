// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log1p.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/log1p.hpp"

namespace ov {
namespace op {

namespace log1p {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        reference::log1p(arg.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace log1p

namespace v16 {

Log1p::Log1p(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Log1p::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v16_Log1p_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Log1p>(new_args.at(0));
}

bool Log1p::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v16_Log1p_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& in_shape = inputs[0].get_shape();
    outputs[0].set_shape(in_shape);

    using namespace ov::element;
    return IF_TYPE_OF(v16_Log1p_evaluate,
                      OV_PP_ET_LIST(bf16, f16, f32, f64),
                      log1p::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      outputs[0],
                      shape_size(inputs[0].get_shape()));
}

bool Log1p::has_evaluate() const {
    OV_OP_SCOPE(v16_Log1p_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}

}  // namespace v16
}  // namespace op
}  // namespace ov
