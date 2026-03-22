// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Log1p operator: log(1 + x).
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void log1p(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, [](const T v) {
        return static_cast<T>(std::log1p(static_cast<double>(v)));
    });
}

}  // namespace reference
}  // namespace ov
