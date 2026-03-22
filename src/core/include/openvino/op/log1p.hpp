// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v16 {
/// \brief Elementwise log1p operation: log(1 + x)
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Log1p : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Log1p", "opset16", util::UnaryElementwiseArithmetic);
    /// \brief Constructs a log1p operation.
    Log1p() = default;
    /// \brief Constructs a log1p operation.
    ///
    /// \param arg Node that produces the input tensor.
    Log1p(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v16
}  // namespace op
}  // namespace ov
