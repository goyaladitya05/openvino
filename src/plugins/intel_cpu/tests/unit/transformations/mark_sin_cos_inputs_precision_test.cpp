// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/mark_sin_cos_inputs_precision.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace ov;

// RoPE-style runtime table computation: positions * freqs -> Sin/Cos -> applied to a MatMul output.
// The angle chain must be marked to stay in f32; the MatMul and the apply-multiply must not.
TEST(MarkSinCosInputsPrecisionTest, MarksAngleChainStopsAtMatMul) {
    auto positions = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 1});
    auto freqs = op::v0::Constant::create(element::f32, Shape{1, 32}, std::vector<float>(32, 100.0F));
    auto angles = std::make_shared<op::v1::Multiply>(positions, freqs);
    auto shifted = std::make_shared<op::v1::Add>(angles, op::v0::Constant::create(element::f32, Shape{}, {-1.0F}));
    auto sin = std::make_shared<op::v0::Sin>(shifted);
    auto cos = std::make_shared<op::v0::Cos>(shifted);

    auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 32});
    auto weights = op::v0::Constant::create(element::f32, Shape{32, 32}, std::vector<float>(32 * 32, 0.01F));
    auto proj = std::make_shared<op::v0::MatMul>(x, weights);

    auto apply_cos = std::make_shared<op::v1::Multiply>(proj, cos);
    auto apply_sin = std::make_shared<op::v1::Multiply>(proj, sin);
    auto sum = std::make_shared<op::v1::Add>(apply_cos, apply_sin);

    auto model = std::make_shared<Model>(OutputVector{sum}, ParameterVector{positions, x});

    ov::pass::Manager manager;
    manager.register_pass<intel_cpu::MarkSinCosInputsPrecision>();
    manager.run_passes(model);

    // the trig ops and their argument chain keep f32
    EXPECT_TRUE(is_conversion_disabled(sin, element::f16));
    EXPECT_TRUE(is_conversion_disabled(cos, element::f16));
    EXPECT_TRUE(is_conversion_disabled(shifted, element::f16));
    EXPECT_TRUE(is_conversion_disabled(angles, element::f16));

    // compute-heavy ops and the downstream apply stay unmarked (low precision preserved)
    EXPECT_FALSE(is_conversion_disabled(proj, element::f16));
    EXPECT_FALSE(is_conversion_disabled(apply_cos, element::f16));
    EXPECT_FALSE(is_conversion_disabled(apply_sin, element::f16));
    EXPECT_FALSE(is_conversion_disabled(sum, element::f16));
}

// The backward walk must not cross a MatMul: sin(matmul(x)) marks only the Sin itself.
TEST(MarkSinCosInputsPrecisionTest, DoesNotCrossMatMul) {
    auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 16});
    auto weights = op::v0::Constant::create(element::f32, Shape{16, 16}, std::vector<float>(16 * 16, 0.01F));
    auto proj = std::make_shared<op::v0::MatMul>(x, weights);
    auto sin = std::make_shared<op::v0::Sin>(proj);
    auto model = std::make_shared<Model>(OutputVector{sin}, ParameterVector{x});

    ov::pass::Manager manager;
    manager.register_pass<intel_cpu::MarkSinCosInputsPrecision>();
    manager.run_passes(model);

    EXPECT_TRUE(is_conversion_disabled(sin, element::f16));
    EXPECT_FALSE(is_conversion_disabled(proj, element::f16));
}

// Integer subpaths (index computation) are left alone; the f32 chain after Convert is marked.
TEST(MarkSinCosInputsPrecisionTest, StopsAtIntegerPath) {
    auto start = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto stop = op::v0::Constant::create(element::i32, Shape{}, {128});
    auto step = op::v0::Constant::create(element::i32, Shape{}, {1});
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto to_f32 = std::make_shared<op::v0::Convert>(range, element::f32);
    auto scaled = std::make_shared<op::v1::Multiply>(to_f32, op::v0::Constant::create(element::f32, Shape{}, {100.0F}));
    auto cos = std::make_shared<op::v0::Cos>(scaled);
    auto model = std::make_shared<Model>(OutputVector{cos}, ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<intel_cpu::MarkSinCosInputsPrecision>();
    manager.run_passes(model);

    EXPECT_TRUE(is_conversion_disabled(cos, element::f16));
    EXPECT_TRUE(is_conversion_disabled(scaled, element::f16));
    EXPECT_TRUE(is_conversion_disabled(to_f32, element::f16));
    EXPECT_FALSE(is_conversion_disabled(range, element::f16));
}
