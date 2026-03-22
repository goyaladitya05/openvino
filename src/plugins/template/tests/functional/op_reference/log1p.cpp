// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log1p.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct Log1pParams {
    template <class IT>
    Log1pParams(const PartialShape& shape,
                const element::Type& iType,
                const std::vector<IT>& iValues,
                const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceLog1pLayerTest : public testing::TestWithParam<Log1pParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<Log1pParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto log1p = std::make_shared<op::v16::Log1p>(in);
        return std::make_shared<Model>(OutputVector{log1p}, ParameterVector{in});
    }
};

TEST_P(ReferenceLog1pLayerTest, Log1pWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<Log1pParams> generateParamsForLog1p() {
    using T = typename element_type_traits<IN_ET>::value_type;

    // log1p(x) = log(1 + x); reference values computed with std::log1p
    std::vector<Log1pParams> params{
        Log1pParams(ov::PartialShape{6},
                    IN_ET,
                    std::vector<T>{0.0f, 0.5f, 1.0f, -0.5f, 3.0f, -0.9f},
                    std::vector<T>{0.0f,
                                   0.405465108f,
                                   0.693147181f,
                                   -0.693147181f,
                                   1.386294361f,
                                   -2.302585093f})};
    return params;
}

std::vector<Log1pParams> generateCombinedParamsForLog1p() {
    const std::vector<std::vector<Log1pParams>> allTypeParams{
        generateParamsForLog1p<element::Type_t::f32>(),
        generateParamsForLog1p<element::Type_t::f16>(),
    };

    std::vector<Log1pParams> combinedParams;
    for (const auto& p : allTypeParams) {
        combinedParams.insert(combinedParams.end(), p.begin(), p.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Log1p_With_Hardcoded_Refs,
                         ReferenceLog1pLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForLog1p()),
                         ReferenceLog1pLayerTest::getTestCaseName);

}  // namespace
