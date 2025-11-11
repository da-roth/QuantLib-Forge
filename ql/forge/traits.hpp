// This file is part of QuantLib-Forge <https://github.com/da-roth/QuantLib-Forge>
//
// Adapter to integrate Forge AAD engine with QuantLib
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file traits.hpp
 * @brief Helper traits and utilities for Forge types
 *
 * Provides type traits and conversion utilities for working with
 * Forge active types in QuantLib context.
 */

#pragma once

#include <type_traits>
#include "../../../forge/tools/types/fdouble.hpp"

namespace qlforge {

    /**
     * @brief Extract passive (double) value from forge::fdouble
     * @param x The Forge active type
     * @return The underlying double value
     */
    inline double to_value(const forge::fdouble& x) {
        return x.value();
    }

    /**
     * @brief Overload for plain double (passthrough)
     */
    inline double to_value(double x) {
        return x;
    }

    /**
     * @brief Type trait to detect Forge active numbers
     */
    template<typename T>
    struct is_forge_number : std::false_type {};

    template<>
    struct is_forge_number<forge::fdouble> : std::true_type {};

    /**
     * @brief Helper variable template (C++17)
     */
    template<typename T>
    inline constexpr bool is_forge_number_v = is_forge_number<T>::value;

} // namespace qlforge
