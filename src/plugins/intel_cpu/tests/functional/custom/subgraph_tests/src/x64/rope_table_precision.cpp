// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Runtime-computed RoPE table applied to a projection, as in models where RoPEFusion does not
// fire (e.g. LTX-Video's 3-axis rope): angles = positions * freqs reach magnitudes of
// theta*pi/2 (~1e4 rad). If the angle computation is lowered to bf16, the ~2^-8 relative
// quantization makes sin/cos of the high-frequency bands meaningless. The angle chain must be
// kept in f32 (MarkSinCosInputsPrecision), while the projection MatMuls stay in bf16.
class RopeTablePrecisionCPUTest : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::bf16});
        rel_threshold = abs_threshold = 0.05;

        const std::vector<InputShape> input_shapes = {
            {{-1, 1}, {{64, 1}, {128, 1}}},                                // positions
            {{-1, HIDDEN_SIZE}, {{64, HIDDEN_SIZE}, {128, HIDDEN_SIZE}}},  // hidden states
        };
        init_input_shapes(input_shapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        // rope table: angles up to positions_max * freqs_max = 128 * 4 ~ 500 rad. Large enough that
        // bf16 (~2 rad quantization at that magnitude) scrambles sin/cos without the f32 marking,
        // yet small enough that f32 sin/cos implementations (CPU JIT vs reference backend) still
        // agree — huge arguments diverge across range-reduction implementations even in f32.
        std::vector<float> freq_values(HIDDEN_SIZE);
        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            freq_values[i] = 0.01F * std::pow(400.0F, static_cast<float>(i) / HIDDEN_SIZE);
        }
        auto freqs = utils::make_constant(ov::element::f32, ov::Shape{1, HIDDEN_SIZE}, freq_values);
        auto angles = std::make_shared<ov::op::v1::Multiply>(params[0], freqs);
        auto cos = std::make_shared<ov::op::v0::Cos>(angles);
        auto sin = std::make_shared<ov::op::v0::Sin>(angles);

        // projection (mandatory bf16)
        auto proj_weights = utils::make_constant(ov::element::f32, ov::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
        auto q = std::make_shared<ov::op::v0::MatMul>(params[1], proj_weights);

        // apply the table
        auto q_cos = std::make_shared<ov::op::v1::Multiply>(q, cos);
        auto q_sin = std::make_shared<ov::op::v1::Multiply>(q, sin);
        auto rotated = std::make_shared<ov::op::v1::Add>(q_cos, q_sin);

        auto out_weights = utils::make_constant(ov::element::f32, ov::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotated, out_weights);

        function = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(matmul)},
                                               params,
                                               "RopeTablePrecision");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        // integer positions 0..N-1, like real position indices
        ov::Tensor positions{ov::element::f32, targetInputStaticShapes[0]};
        auto* pos_data = positions.data<float>();
        for (size_t i = 0; i < positions.get_size(); ++i) {
            pos_data[i] = static_cast<float>(i);
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), positions});

        utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 256;
        auto hidden =
            utils::create_and_fill_tensor(funcInputs[1].get_element_type(), targetInputStaticShapes[1], in_data);
        inputs.insert({funcInputs[1].get_node_shared_ptr(), hidden});
    }

    static constexpr size_t HIDDEN_SIZE = 64;
};

TEST_F(RopeTablePrecisionCPUTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "No bf16 hardware support";
    }
    run();
}

}  // namespace ov::test
