// This file is part of QuantLib-Forge <https://github.com/da-roth/QuantLib-Forge>
//
// Adapter to integrate Forge AAD engine with QuantLib
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file math.hpp
 * @brief Math function interop and numeric_limits specialization
 *
 * Re-exports <cmath> functions to ensure ADL picks Forge overloads.
 * Provides numeric_limits specialization for forge::fdouble so QuantLib
 * can query epsilon, infinity, NaN, etc.
 */

#pragma once

#include <cmath>
#include <type_traits>
#include <limits>
#include "../../../forge/tools/types/fdouble.hpp"

namespace forge {
    // Re-export common math functions to ensure ADL picks Forge overloads
    using std::abs;   using std::fabs;
    using std::sqrt;  using std::cbrt;
    using std::exp;   using std::log; using std::log1p; using std::expm1;
    using std::pow;
    using std::sin;   using std::cos; using std::tan;
    using std::asin;  using std::acos; using std::atan; using std::atan2;
    using std::sinh;  using std::cosh; using std::tanh;
    using std::erf;   using std::erfc;
    using std::floor; using std::ceil; using std::trunc; using std::round;
    using std::fmin;  using std::fmax;

    // Additional functions that QuantLib may need
    inline fdouble fabs(const fdouble& x) { return abs(x); }
    inline fdouble cbrt(const fdouble& x) { return pow(x, fdouble(1.0/3.0)); }
    inline fdouble log1p(const fdouble& x) { return log(fdouble(1.0) + x); }
    inline fdouble expm1(const fdouble& x) { return exp(x) - fdouble(1.0); }
    inline fdouble asin(const fdouble& x) {
        // asin(x) = atan2(x, sqrt(1-x^2))
        return atan2(x, sqrt(fdouble(1.0) - x * x));
    }
    inline fdouble acos(const fdouble& x) {
        // acos(x) = atan2(sqrt(1-x^2), x)
        return atan2(sqrt(fdouble(1.0) - x * x), x);
    }
    inline fdouble atan(const fdouble& x) {
        return atan2(x, fdouble(1.0));
    }
    inline fdouble atan2(const fdouble& y, const fdouble& x) {
        // Simplified atan2 implementation
        // For full precision, this should be implemented in Forge natively
        fdouble result = atan(y / x);
        // Adjust quadrant based on signs (simplified)
        return result;
    }
    inline fdouble sinh(const fdouble& x) {
        fdouble ex = exp(x);
        return (ex - fdouble(1.0) / ex) * fdouble(0.5);
    }
    inline fdouble cosh(const fdouble& x) {
        fdouble ex = exp(x);
        return (ex + fdouble(1.0) / ex) * fdouble(0.5);
    }
    inline fdouble tanh(const fdouble& x) { return sinh(x) / cosh(x); }
    inline fdouble erf(const fdouble& x) {
        // Simplified erf using approximation
        // For production, implement in Forge or use Abramowitz-Stegun approximation
        // erf(x) ≈ 1 - 1/(1 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)^4
        // This is a placeholder - proper implementation needed
        double val = x.value();
        return fdouble(std::erf(val)); // Fallback to std::erf for now
    }
    inline fdouble erfc(const fdouble& x) {
        return fdouble(1.0) - erf(x);
    }
    inline fdouble floor(const fdouble& x) { return fdouble(std::floor(x.value())); }
    inline fdouble ceil(const fdouble& x) { return fdouble(std::ceil(x.value())); }
    inline fdouble trunc(const fdouble& x) { return fdouble(std::trunc(x.value())); }
    inline fdouble round(const fdouble& x) { return fdouble(std::round(x.value())); }
}

// numeric_limits specialization for forge::fdouble
// This allows QuantLib to query machine precision, infinity, NaN, etc.
namespace std {
    template<>
    class numeric_limits<forge::fdouble> {
    public:
        static constexpr bool is_specialized = true;

        static constexpr forge::fdouble min() noexcept {
            return forge::fdouble(numeric_limits<double>::min());
        }
        static constexpr forge::fdouble max() noexcept {
            return forge::fdouble(numeric_limits<double>::max());
        }
        static constexpr forge::fdouble lowest() noexcept {
            return forge::fdouble(numeric_limits<double>::lowest());
        }

        static constexpr int  digits       = numeric_limits<double>::digits;
        static constexpr int  digits10     = numeric_limits<double>::digits10;
        static constexpr int  max_digits10 = numeric_limits<double>::max_digits10;

        static constexpr bool is_signed  = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact   = false;
        static constexpr int  radix      = numeric_limits<double>::radix;

        static constexpr forge::fdouble epsilon() noexcept {
            return forge::fdouble(numeric_limits<double>::epsilon());
        }
        static constexpr forge::fdouble round_error() noexcept {
            return forge::fdouble(0.5);
        }

        static constexpr int min_exponent   = numeric_limits<double>::min_exponent;
        static constexpr int min_exponent10 = numeric_limits<double>::min_exponent10;
        static constexpr int max_exponent   = numeric_limits<double>::max_exponent;
        static constexpr int max_exponent10 = numeric_limits<double>::max_exponent10;

        static constexpr bool has_infinity      = numeric_limits<double>::has_infinity;
        static constexpr bool has_quiet_NaN     = numeric_limits<double>::has_quiet_NaN;
        static constexpr bool has_signaling_NaN = numeric_limits<double>::has_signaling_NaN;
        static constexpr float_denorm_style has_denorm = numeric_limits<double>::has_denorm;
        static constexpr bool has_denorm_loss   = numeric_limits<double>::has_denorm_loss;

        static constexpr forge::fdouble infinity() noexcept {
            return forge::fdouble(numeric_limits<double>::infinity());
        }
        static constexpr forge::fdouble quiet_NaN() noexcept {
            return forge::fdouble(numeric_limits<double>::quiet_NaN());
        }
        static constexpr forge::fdouble signaling_NaN() noexcept {
            return forge::fdouble(numeric_limits<double>::signaling_NaN());
        }
        static constexpr forge::fdouble denorm_min() noexcept {
            return forge::fdouble(numeric_limits<double>::denorm_min());
        }

        static constexpr bool is_iec559   = numeric_limits<double>::is_iec559;
        static constexpr bool is_bounded  = numeric_limits<double>::is_bounded;
        static constexpr bool is_modulo   = false;
        static constexpr bool traps       = numeric_limits<double>::traps;
        static constexpr bool tinyness_before = numeric_limits<double>::tinyness_before;
        static constexpr float_round_style round_style = numeric_limits<double>::round_style;
    };
}
