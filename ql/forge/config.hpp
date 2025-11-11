// This file is part of QuantLib-Forge <https://github.com/da-roth/QuantLib-Forge>
//
// Adapter to integrate Forge AAD engine with QuantLib
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file config.hpp
 * @brief Type substitution: QuantLib::Real = forge::fdouble
 *
 * This header must be included before QuantLib headers to swap the Real type.
 * It allows QuantLib to record computation graphs for automatic differentiation.
 */

#pragma once

#include <limits>
#include <cmath>

// Include Forge public headers for fdouble and recording
#include "../../../forge/tools/types/fdouble.hpp"

namespace QuantLib {
    // Central substitution: replace QuantLib's Real with Forge's active type
    using Real = forge::fdouble;
}
