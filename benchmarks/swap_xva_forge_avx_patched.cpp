/*******************************************************************************

   XVA Performance Benchmark - Forge AVX2 Comparison Version (Patched)

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
   Original QuantLib code: Copyright (C) 2003-2007 Ferdinando Ametrano,
                           StatPro Italia srl, Joseph Wang

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

   NOTE: This version requires patched Forge with optimized buffer access methods:
         - getBufferIndex()
         - setVectorValueDirect()
         - getVectorValueDirect()
         - getGradientsDirectLane()

******************************************************************************/

// =============================================================================
// XVA PERFORMANCE TEST - FORGE SSE2 vs AVX2 COMPARISON (PATCHED VERSION)
// =============================================================================
// This test compares FOUR approaches:
//   1. Bump-Reval (baseline)        - Direct QuantLib evaluation with finite differences
//   2. Forge Kernel Bump-Reval      - JIT-compiled kernel (SSE2), still using bump-reval
//   3. Forge AAD (SSE2)             - JIT-compiled kernel with AAD (scalar SSE2)
//   4. Forge AAD (AVX2)             - JIT-compiled kernel with AAD (4-wide AVX2)
//
// The AVX2 version processes 4 doubles per operation using YMM registers,
// which should provide additional speedup for the forward pass computation.
// =============================================================================

#include <ql/qldefines.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/settings.hpp>

// Forge integration headers
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/compiler_config.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h>  // AVX2 intrinsics for direct buffer writes

// Check for AVX2 support at runtime
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

using namespace QuantLib;

namespace {

    //=========================================================================
    // Check if AVX2 is supported on this CPU
    //=========================================================================
    bool isAVX2Supported() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        int nIds = cpuInfo[0];
        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            return (cpuInfo[1] & (1 << 5)) != 0; // AVX2 bit
        }
        return false;
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            return (ebx & (1 << 5)) != 0; // AVX2 bit
        }
        return false;
#endif
#else
        return false; // Non-x86 architecture
#endif
    }

    //=========================================================================
    // IR Curve Pillar Definition
    //=========================================================================
    struct IRPillar {
        std::string name;
        Period tenor;
        double baseRate;
        double volatility;
    };

    //=========================================================================
    // Standard IR curve pillars (typical XVA setup)
    //=========================================================================
    std::vector<IRPillar> createEURCurvePillars() {
        return {
            {"EUR_6M",  Period(6, Months),  0.0320, 0.0060},
            {"EUR_1Y",  Period(1, Years),   0.0335, 0.0055},
            {"EUR_2Y",  Period(2, Years),   0.0348, 0.0050},
            {"EUR_3Y",  Period(3, Years),   0.0358, 0.0048},
            {"EUR_5Y",  Period(5, Years),   0.0375, 0.0045},
            {"EUR_7Y",  Period(7, Years),   0.0388, 0.0043},
            {"EUR_10Y", Period(10, Years),  0.0402, 0.0040},
            {"EUR_15Y", Period(15, Years),  0.0415, 0.0038},
            {"EUR_20Y", Period(20, Years),  0.0422, 0.0035},
            {"EUR_30Y", Period(30, Years),  0.0428, 0.0032}
        };
    }

    //=========================================================================
    // Additional risk factors for XVA (FX, credit, other curves)
    //=========================================================================
    struct AdditionalRiskFactors {
        std::vector<double> usdRates;
        std::vector<double> gbpRates;
        std::vector<double> jpyRates;
        std::vector<double> chfRates;
        std::vector<double> fxRates;
        std::vector<double> counterpartySpreads;
        std::vector<double> ownSpreads;
        std::vector<double> volSurface;
    };

    AdditionalRiskFactors createBaseRiskFactors() {
        AdditionalRiskFactors rf;
        rf.usdRates = {0.0480, 0.0495, 0.0505, 0.0512, 0.0525, 0.0535, 0.0545, 0.0555, 0.0560, 0.0565};
        rf.gbpRates = {0.0420, 0.0435, 0.0448, 0.0458, 0.0470, 0.0480, 0.0490, 0.0500, 0.0505, 0.0510};
        rf.jpyRates = {-0.001, 0.000, 0.002, 0.004, 0.008, 0.012, 0.018, 0.022, 0.025, 0.028};
        rf.chfRates = {0.010, 0.012, 0.015, 0.018, 0.022, 0.026, 0.030, 0.034, 0.037, 0.040};
        rf.fxRates = {1.08, 1.27, 149.5, 0.88, 0.85};
        rf.counterpartySpreads = {0.0050, 0.0055, 0.0062, 0.0070, 0.0080, 0.0092, 0.0105, 0.0120, 0.0135, 0.0150};
        rf.ownSpreads = {0.0030, 0.0033, 0.0038, 0.0044, 0.0052, 0.0060, 0.0070, 0.0082, 0.0095, 0.0110};
        rf.volSurface = {
            0.20, 0.19, 0.18, 0.19, 0.21,
            0.19, 0.18, 0.17, 0.18, 0.20,
            0.18, 0.17, 0.16, 0.17, 0.19,
            0.17, 0.16, 0.15, 0.16, 0.18,
            0.16, 0.15, 0.14, 0.15, 0.17
        };
        return rf;
    }

    //=========================================================================
    // Configuration
    //=========================================================================
    struct XvaConfig {
        std::string name;
        Size numSwaps = 1;
        Size numTimeSteps = 1;
        Size numPaths = 1;
        Size numRiskFactors = 10;
        Size warmupRuns = 2;
        Size timedRuns = 5;
        double bumpSize = 1e-4;
    };

    //=========================================================================
    // Swap Definition
    //=========================================================================
    struct SwapDefinition {
        Integer tenorYears;
        Period fixedFreq;
        Period floatFreq;
        Real notional;
        Rate fixedRate;
        Real spread;
    };

    //=========================================================================
    // Market Scenario
    //=========================================================================
    struct MarketScenario {
        std::vector<double> flatData;

        const std::vector<double>& flatten() const { return flatData; }
    };

    //=========================================================================
    // Results
    //=========================================================================
    struct XvaResults {
        std::vector<std::vector<std::vector<double>>> exposures;
        std::vector<std::vector<std::vector<std::vector<double>>>> sensitivities;
        double expectedExposure = 0.0;
        double cva = 0.0;
    };

    //=========================================================================
    // Optimization Mode for Forge compiler
    //=========================================================================
    enum class OptimizationMode {
        Default,        // All optimizations ON
        StabilityOnly,  // All OFF except enableStabilityCleaning = true
        NoOpt           // All optimizations OFF
    };

    // Helper to configure CompilerConfig based on optimization mode
    forge::CompilerConfig configureOptimizations(OptimizationMode mode) {
        forge::CompilerConfig config;
        switch (mode) {
            case OptimizationMode::Default:
                // Explicitly enable all optimizations
                config.enableOptimizations = true;
                config.enableCSE = true;
                config.enableAlgebraicSimplification = true;
                config.enableInactiveFolding = true;
                config.enableStabilityCleaning = true;
                break;
            case OptimizationMode::StabilityOnly:
                // Use default (which is now stability-only)
                config = forge::CompilerConfig::Default();
                break;
            case OptimizationMode::NoOpt:
                config = forge::CompilerConfig::NoOptimization();
                break;
        }
        return config;
    }

    //=========================================================================
    // Timing Results
    //=========================================================================
    struct TimingResults {
        double totalTimeMs = 0.0;
        double avgTimePerIterationMs = 0.0;
        double kernelCreationTimeMs = 0.0;
        double evaluationTimeMs = 0.0;        // Total eval time (sum of below)
        double setInputsTimeMs = 0.0;         // Time spent setting inputs
        double executeKernelTimeMs = 0.0;     // Time spent executing kernel
        double getOutputsTimeMs = 0.0;        // Time spent getting outputs
        double getGradientsTimeMs = 0.0;      // Time spent getting gradients
        double singleScenarioTimeUs = 0.0;
        Size numScenarios = 0;
        Size numEvaluations = 0;
        Size numKernelsCreated = 0;
    };

    //=========================================================================
    // Generate scenarios with specified number of risk factors
    //=========================================================================
    std::vector<std::vector<MarketScenario>> generateScenarios(
        const XvaConfig& config,
        const std::vector<IRPillar>& eurPillars,
        const AdditionalRiskFactors& baseFactors,
        unsigned int seed = 42) {

        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0.0, 1.0);

        std::vector<std::vector<MarketScenario>> scenarios(config.numTimeSteps);

        double rateVol = 0.005;
        double fxVol = 0.10;
        double creditVol = 0.20;
        double volVol = 0.30;

        for (Size t = 0; t < config.numTimeSteps; ++t) {
            scenarios[t].resize(config.numPaths);
            double timeYears = config.numTimeSteps > 1 ? double(t + 1) / config.numTimeSteps * 5.0 : 0.0;
            double sqrtTime = std::sqrt(std::max(timeYears, 0.01));

            for (Size p = 0; p < config.numPaths; ++p) {
                MarketScenario& sc = scenarios[t][p];
                sc.flatData.resize(config.numRiskFactors);
                Size idx = 0;

                // EUR curve (first 10 risk factors)
                double eurParallel = dist(gen);
                for (Size i = 0; i < std::min(Size(10), config.numRiskFactors); ++i) {
                    sc.flatData[idx++] = eurPillars[i].baseRate + eurPillars[i].volatility * eurParallel * sqrtTime;
                }

                if (config.numRiskFactors <= 10) continue;

                // USD curve (10-19)
                double usdParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.usdRates[i] + rateVol * usdParallel * sqrtTime;
                }

                // GBP curve (20-29)
                double gbpParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.gbpRates[i] + rateVol * gbpParallel * sqrtTime;
                }

                // JPY curve (30-39)
                double jpyParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.jpyRates[i] + rateVol * jpyParallel * sqrtTime;
                }

                // CHF curve (40-49)
                double chfParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.chfRates[i] + rateVol * chfParallel * sqrtTime;
                }

                // FX rates (50-54)
                for (Size i = 0; i < 5 && idx < config.numRiskFactors; ++i) {
                    double fxShock = dist(gen);
                    sc.flatData[idx++] = baseFactors.fxRates[i] * std::exp(fxVol * fxShock * sqrtTime - 0.5 * fxVol * fxVol * timeYears);
                }

                // Counterparty credit spreads (55-64)
                double cptyShock = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.counterpartySpreads[i] * std::exp(creditVol * cptyShock * sqrtTime);
                }

                // Own credit spreads (65-74)
                double ownShock = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.ownSpreads[i] * std::exp(creditVol * ownShock * sqrtTime);
                }

                // Vol surface (75-99)
                double volShock = dist(gen);
                for (Size i = 0; i < 25 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.volSurface[i] * std::exp(volVol * volShock * sqrtTime);
                }
            }
        }
        return scenarios;
    }

    //=========================================================================
    // Create swap definitions
    //=========================================================================
    std::vector<SwapDefinition> createSwapDefinitions(Size numSwaps) {
        std::vector<SwapDefinition> swaps;
        swaps.push_back({5, Period(1, Years), Period(6, Months), 1000000.0, 0.03, 0.001});
        if (numSwaps > 1) {
            swaps.push_back({10, Period(6, Months), Period(3, Months), 2000000.0, 0.035, 0.0015});
        }
        return swaps;
    }

    //=========================================================================
    // Price swap - 10 risk factors (EUR curve only)
    //=========================================================================
    Real priceSwap10RF(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<Real>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter) {

        double timeStepFraction = totalTimeSteps > 1 ? double(timeStep) / totalTimeSteps : 0.0;
        Integer elapsedYears = Integer(timeStepFraction * swapDef.tenorYears);
        Integer remainingYears = swapDef.tenorYears - elapsedYears;

        if (remainingYears <= 0) {
            return Real(0.0);
        }

        std::vector<Date> curveDates;
        std::vector<Real> curveRates;
        curveDates.push_back(today);
        curveRates.push_back(allInputs[0]);

        for (Size i = 0; i < 10 && i < eurPillarDefs.size(); ++i) {
            curveDates.push_back(calendar.advance(today, eurPillarDefs[i].tenor));
            curveRates.push_back(allInputs[i]);
        }

        RelinkableHandle<YieldTermStructure> termStructure;
        auto zeroCurve = ext::make_shared<ZeroCurve>(curveDates, curveRates, dayCounter);
        zeroCurve->enableExtrapolation();
        termStructure.linkTo(zeroCurve);

        auto index = ext::make_shared<Euribor6M>(termStructure);
        Date start = calendar.advance(today, index->fixingDays(), Days);
        Date maturity = calendar.advance(start, remainingYears, Years);

        Schedule fixedSchedule(start, maturity, swapDef.fixedFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);
        Schedule floatSchedule(start, maturity, swapDef.floatFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);

        auto swap = ext::make_shared<VanillaSwap>(
            VanillaSwap::Payer, swapDef.notional, fixedSchedule, swapDef.fixedRate,
            dayCounter, floatSchedule, index, swapDef.spread, dayCounter);

        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(termStructure));
        return swap->NPV();
    }

    //=========================================================================
    // Price swap - 100 risk factors (full XVA)
    //=========================================================================
    Real priceSwap100RF(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<Real>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter) {

        double timeStepFraction = totalTimeSteps > 1 ? double(timeStep) / totalTimeSteps : 0.0;
        Integer elapsedYears = Integer(timeStepFraction * swapDef.tenorYears);
        Integer remainingYears = swapDef.tenorYears - elapsedYears;

        if (remainingYears <= 0) {
            return Real(0.0);
        }

        std::vector<Date> curveDates;
        std::vector<Real> curveRates;
        curveDates.push_back(today);
        curveRates.push_back(allInputs[0]);

        for (Size i = 0; i < 10 && i < eurPillarDefs.size(); ++i) {
            curveDates.push_back(calendar.advance(today, eurPillarDefs[i].tenor));
            curveRates.push_back(allInputs[i]);
        }

        RelinkableHandle<YieldTermStructure> termStructure;
        auto zeroCurve = ext::make_shared<ZeroCurve>(curveDates, curveRates, dayCounter);
        zeroCurve->enableExtrapolation();
        termStructure.linkTo(zeroCurve);

        auto index = ext::make_shared<Euribor6M>(termStructure);
        Date start = calendar.advance(today, index->fixingDays(), Days);
        Date maturity = calendar.advance(start, remainingYears, Years);

        Schedule fixedSchedule(start, maturity, swapDef.fixedFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);
        Schedule floatSchedule(start, maturity, swapDef.floatFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);

        auto swap = ext::make_shared<VanillaSwap>(
            VanillaSwap::Payer, swapDef.notional, fixedSchedule, swapDef.fixedRate,
            dayCounter, floatSchedule, index, swapDef.spread, dayCounter);

        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(termStructure));
        Real baseNpv = swap->NPV();

        // XVA adjustments using remaining 90 risk factors
        std::vector<Real> eurRates(allInputs.begin(), allInputs.begin() + 10);
        std::vector<Real> usdRates(allInputs.begin() + 10, allInputs.begin() + 20);
        std::vector<Real> gbpRates(allInputs.begin() + 20, allInputs.begin() + 30);
        std::vector<Real> jpyRates(allInputs.begin() + 30, allInputs.begin() + 40);
        std::vector<Real> chfRates(allInputs.begin() + 40, allInputs.begin() + 50);
        std::vector<Real> fxRates(allInputs.begin() + 50, allInputs.begin() + 55);
        std::vector<Real> counterpartySpreads(allInputs.begin() + 55, allInputs.begin() + 65);
        std::vector<Real> ownSpreads(allInputs.begin() + 65, allInputs.begin() + 75);
        std::vector<Real> volSurface(allInputs.begin() + 75, allInputs.begin() + 100);

        Real ccyBasisAdj = Real(0.0);
        Real eurusd = fxRates[0];
        Real eurgbp = fxRates[4];
        for (Size i = 0; i < 10; ++i) {
            ccyBasisAdj += (usdRates[i] - eurRates[i]) * eurusd * Real(0.0001) * swapDef.notional;
            ccyBasisAdj += (gbpRates[i] - eurRates[i]) * eurgbp * Real(0.00005) * swapDef.notional;
            ccyBasisAdj += jpyRates[i] * Real(0.00001) * swapDef.notional;
            ccyBasisAdj += chfRates[i] * Real(0.00002) * swapDef.notional;
        }

        Real lgd = Real(0.4);
        Real exposure = baseNpv > Real(0.0) ? baseNpv : Real(0.0);
        Real negExposure = baseNpv < Real(0.0) ? -baseNpv : Real(0.0);

        Real cvaAdj = Real(0.0);
        Real dvaAdj = Real(0.0);
        for (Size i = 0; i < 10; ++i) {
            cvaAdj -= lgd * exposure * counterpartySpreads[i] * Real(0.1);
            dvaAdj += lgd * negExposure * ownSpreads[i] * Real(0.1);
        }

        Real volAdj = Real(0.0);
        for (Size i = 0; i < 25; ++i) {
            volAdj += volSurface[i] * Real(0.001) * swapDef.notional;
        }

        return baseNpv + ccyBasisAdj + cvaAdj + dvaAdj + volAdj;
    }

    //=========================================================================
    // Price swap dispatcher based on number of risk factors
    //=========================================================================
    Real priceSwap(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<Real>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        Size numRiskFactors) {

        if (numRiskFactors <= 10) {
            return priceSwap10RF(swapDef, timeStep, totalTimeSteps, allInputs, eurPillarDefs, today, calendar, dayCounter);
        } else {
            return priceSwap100RF(swapDef, timeStep, totalTimeSteps, allInputs, eurPillarDefs, today, calendar, dayCounter);
        }
    }

    //=========================================================================
    // 1. BUMP-REVAL COMPUTATION (Baseline)
    //=========================================================================
    XvaResults computeBumpReval(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {

        XvaResults results;
        results.exposures.resize(config.numSwaps);
        results.sensitivities.resize(config.numSwaps);

        double totalExposure = 0.0;
        Size totalEvaluations = 0;
        Size numScenarios = 0;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (Size s = 0; s < config.numSwaps; ++s) {
            results.exposures[s].resize(config.numTimeSteps);
            results.sensitivities[s].resize(config.numTimeSteps);

            for (Size t = 0; t < config.numTimeSteps; ++t) {
                results.exposures[s][t].resize(config.numPaths);
                results.sensitivities[s][t].resize(config.numPaths);

                for (Size p = 0; p < config.numPaths; ++p) {
                    const auto& scenario = scenarios[t][p];
                    std::vector<double> flatInputs = scenario.flatten();
                    std::vector<Real> realInputs(flatInputs.begin(), flatInputs.end());

                    double baseNpv = value(priceSwap(
                        swaps[s], t, config.numTimeSteps, realInputs, pillars, today, calendar, dayCounter, config.numRiskFactors));
                    totalEvaluations++;

                    results.exposures[s][t][p] = std::max(0.0, baseNpv);
                    totalExposure += results.exposures[s][t][p];

                    results.sensitivities[s][t][p].resize(config.numRiskFactors);
                    for (Size i = 0; i < config.numRiskFactors; ++i) {
                        std::vector<Real> bumpedInputs = realInputs;
                        bumpedInputs[i] += config.bumpSize;
                        double bumpedNpv = value(priceSwap(
                            swaps[s], t, config.numTimeSteps, bumpedInputs, pillars, today, calendar, dayCounter, config.numRiskFactors));
                        totalEvaluations++;
                        results.sensitivities[s][t][p][i] = (bumpedNpv - baseNpv) / config.bumpSize;
                    }
                    numScenarios++;
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        timing.numScenarios = numScenarios;
        timing.numEvaluations = totalEvaluations;
        timing.evaluationTimeMs = duration.count() / 1000.0;
        timing.singleScenarioTimeUs = numScenarios > 0 ? double(duration.count()) / numScenarios : 0.0;
        timing.numKernelsCreated = 0;
        timing.kernelCreationTimeMs = 0.0;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalScenarios > 0 ? totalExposure / totalScenarios : 0.0;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
    }

    //=========================================================================
    // 2. FORGE AAD COMPUTATION - SSE2 (Scalar, 1 scenario per execution)
    //=========================================================================
    XvaResults computeForgeAadSSE2Impl(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing,
        OptimizationMode optMode = OptimizationMode::Default) {

        XvaResults results;
        results.exposures.resize(config.numSwaps);
        results.sensitivities.resize(config.numSwaps);

        double totalExposure = 0.0;
        Size numKernels = 0;
        Size numEvaluations = 0;
        Size numScenarios = 0;
        double totalKernelCreationUs = 0.0;
        double totalEvalUs = 0.0;
        double totalSetInputsUs = 0.0;
        double totalExecuteKernelUs = 0.0;
        double totalGetOutputsUs = 0.0;
        double totalGetGradientsUs = 0.0;

        for (Size s = 0; s < config.numSwaps; ++s) {
            results.exposures[s].resize(config.numTimeSteps);
            results.sensitivities[s].resize(config.numTimeSteps);

            for (Size t = 0; t < config.numTimeSteps; ++t) {
                results.exposures[s][t].resize(config.numPaths);
                results.sensitivities[s][t].resize(config.numPaths);

                // --- KERNEL CREATION ---
                auto kernelStartTime = std::chrono::high_resolution_clock::now();

                forge::GraphRecorder recorder;
                recorder.start();

                std::vector<double> flatInputs = scenarios[t][0].flatten();
                std::vector<Real> rateInputs(config.numRiskFactors);
                std::vector<forge::NodeId> rateNodeIds(config.numRiskFactors);
                for (Size i = 0; i < config.numRiskFactors; ++i) {
                    rateInputs[i] = flatInputs[i];
                    rateInputs[i].markForgeInputAndDiff();
                    rateNodeIds[i] = rateInputs[i].forgeNodeId();
                }

                Real npv = priceSwap(swaps[s], t, config.numTimeSteps, rateInputs, pillars, today, calendar, dayCounter, config.numRiskFactors);
                npv.markForgeOutput();
                forge::NodeId npvNodeId = npv.forgeNodeId();

                recorder.stop();
                forge::Graph graph = recorder.graph();

                // Configure the compiler for SSE2 scalar
                forge::CompilerConfig compilerConfig = configureOptimizations(optMode);
                compilerConfig.instructionSet = forge::CompilerConfig::InstructionSet::SSE2_SCALAR;

                forge::ForgeEngine compiler(compilerConfig);
                auto kernel = compiler.compile(graph);
                auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

                // Pre-compute gradient buffer indices (vectorWidth=1 for SSE2)
                int vectorWidth = buffer->getVectorWidth();
                std::vector<size_t> gradientIndices(config.numRiskFactors);
                for (Size i = 0; i < config.numRiskFactors; ++i) {
                    gradientIndices[i] = static_cast<size_t>(rateNodeIds[i]) * vectorWidth;
                }

                auto kernelEndTime = std::chrono::high_resolution_clock::now();
                totalKernelCreationUs += std::chrono::duration_cast<std::chrono::microseconds>(kernelEndTime - kernelStartTime).count();
                numKernels++;

                // --- EVALUATION (1 scenario per execution) ---
                auto evalStartTime = std::chrono::high_resolution_clock::now();

                std::vector<double> gradOutput(config.numRiskFactors);

                for (Size p = 0; p < config.numPaths; ++p) {
                    const auto& scenario = scenarios[t][p];
                    std::vector<double> scenarioInputs = scenario.flatten();

                    // Set inputs (scalar - one value per node)
                    auto setInputsStart = std::chrono::high_resolution_clock::now();
                    for (Size i = 0; i < config.numRiskFactors; ++i) {
                        buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                    }
                    auto setInputsEnd = std::chrono::high_resolution_clock::now();
                    totalSetInputsUs += std::chrono::duration_cast<std::chrono::nanoseconds>(setInputsEnd - setInputsStart).count() / 1000.0;

                    // Execute kernel
                    auto executeStart = std::chrono::high_resolution_clock::now();
                    kernel->execute(*buffer);
                    auto executeEnd = std::chrono::high_resolution_clock::now();
                    totalExecuteKernelUs += std::chrono::duration_cast<std::chrono::nanoseconds>(executeEnd - executeStart).count() / 1000.0;
                    numEvaluations++;

                    // Get outputs
                    auto getOutputsStart = std::chrono::high_resolution_clock::now();
                    double npvValue = buffer->getValue(npvNodeId);
                    results.exposures[s][t][p] = std::max(0.0, npvValue);
                    totalExposure += results.exposures[s][t][p];
                    auto getOutputsEnd = std::chrono::high_resolution_clock::now();
                    totalGetOutputsUs += std::chrono::duration_cast<std::chrono::nanoseconds>(getOutputsEnd - getOutputsStart).count() / 1000.0;

                    // Get gradients
                    auto getGradientsStart = std::chrono::high_resolution_clock::now();
                    buffer->getGradientsDirect(gradientIndices, gradOutput.data());
                    results.sensitivities[s][t][p] = gradOutput;
                    auto getGradientsEnd = std::chrono::high_resolution_clock::now();
                    totalGetGradientsUs += std::chrono::duration_cast<std::chrono::nanoseconds>(getGradientsEnd - getGradientsStart).count() / 1000.0;

                    numScenarios++;
                }

                auto evalEndTime = std::chrono::high_resolution_clock::now();
                totalEvalUs += std::chrono::duration_cast<std::chrono::microseconds>(evalEndTime - evalStartTime).count();
            }
        }

        timing.numKernelsCreated = numKernels;
        timing.numEvaluations = numEvaluations;
        timing.numScenarios = numScenarios;
        timing.kernelCreationTimeMs = totalKernelCreationUs / 1000.0;
        timing.evaluationTimeMs = totalEvalUs / 1000.0;
        timing.setInputsTimeMs = totalSetInputsUs / 1000.0;
        timing.executeKernelTimeMs = totalExecuteKernelUs / 1000.0;
        timing.getOutputsTimeMs = totalGetOutputsUs / 1000.0;
        timing.getGradientsTimeMs = totalGetGradientsUs / 1000.0;
        timing.singleScenarioTimeUs = numScenarios > 0 ? totalEvalUs / numScenarios : 0.0;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalScenarios > 0 ? totalExposure / totalScenarios : 0.0;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
    }

    //=========================================================================
    // 3. FORGE AAD COMPUTATION - AVX2 (4-wide SIMD, 4 scenarios per execution)
    //    Uses PATCHED Forge with optimized buffer access methods
    //=========================================================================
    XvaResults computeForgeAadAVX2Impl(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing,
        OptimizationMode optMode = OptimizationMode::Default) {

        XvaResults results;
        results.exposures.resize(config.numSwaps);
        results.sensitivities.resize(config.numSwaps);

        double totalExposure = 0.0;
        Size numKernels = 0;
        Size numEvaluations = 0;
        Size numScenarios = 0;
        double totalKernelCreationUs = 0.0;
        double totalEvalUs = 0.0;
        double totalSetInputsUs = 0.0;
        double totalExecuteKernelUs = 0.0;
        double totalGetOutputsUs = 0.0;
        double totalGetGradientsUs = 0.0;

        const int VECTOR_WIDTH = 4;  // AVX2 processes 4 doubles at a time

        for (Size s = 0; s < config.numSwaps; ++s) {
            results.exposures[s].resize(config.numTimeSteps);
            results.sensitivities[s].resize(config.numTimeSteps);

            for (Size t = 0; t < config.numTimeSteps; ++t) {
                results.exposures[s][t].resize(config.numPaths);
                results.sensitivities[s][t].resize(config.numPaths);

                // Pre-allocate sensitivities for all paths to avoid allocation in hot loop
                for (Size p = 0; p < config.numPaths; ++p) {
                    results.sensitivities[s][t][p].resize(config.numRiskFactors);
                }

                // --- KERNEL CREATION ---
                auto kernelStartTime = std::chrono::high_resolution_clock::now();

                forge::GraphRecorder recorder;
                recorder.start();

                std::vector<double> flatInputs = scenarios[t][0].flatten();
                std::vector<Real> rateInputs(config.numRiskFactors);
                std::vector<forge::NodeId> rateNodeIds(config.numRiskFactors);
                for (Size i = 0; i < config.numRiskFactors; ++i) {
                    rateInputs[i] = flatInputs[i];
                    rateInputs[i].markForgeInputAndDiff();
                    rateNodeIds[i] = rateInputs[i].forgeNodeId();
                }

                Real npv = priceSwap(swaps[s], t, config.numTimeSteps, rateInputs, pillars, today, calendar, dayCounter, config.numRiskFactors);
                npv.markForgeOutput();
                forge::NodeId npvNodeId = npv.forgeNodeId();

                recorder.stop();
                forge::Graph graph = recorder.graph();

                // Configure the compiler for AVX2 packed (4-wide)
                forge::CompilerConfig compilerConfig = configureOptimizations(optMode);
                compilerConfig.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;

                forge::ForgeEngine compiler(compilerConfig);
                auto kernel = compiler.compile(graph);
                auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

                // Pre-compute buffer indices for optimized access (no mapping in hot loop)
                // NOTE: These methods require PATCHED Forge
                std::vector<size_t> gradientIndices(config.numRiskFactors);
                std::vector<size_t> inputIndices(config.numRiskFactors);
                bool allIndicesValid = true;
                Size invalidCount = 0;
                for (Size i = 0; i < config.numRiskFactors; ++i) {
                    gradientIndices[i] = buffer->getBufferIndex(rateNodeIds[i]);
                    inputIndices[i] = gradientIndices[i];  // Same indices for inputs
                    if (gradientIndices[i] == SIZE_MAX) {
                        allIndicesValid = false;
                        invalidCount++;
                    }
                }
                size_t outputIndex = buffer->getBufferIndex(npvNodeId);
                if (outputIndex == SIZE_MAX) {
                    allIndicesValid = false;
                    invalidCount++;
                }

                // Report once per kernel if falling back to vanilla API
                static bool reportedFallback = false;
                if (!allIndicesValid && !reportedFallback) {
                    std::cerr << "[AVX2 PATCHED] Warning: " << invalidCount
                              << " invalid buffer indices, using fallback API" << std::endl;
                    reportedFallback = true;
                }

                // Get raw buffer pointer for direct memory access (Option 6 optimization)
                double* valuesPtr = buffer->getValuesPtr();
                double* gradientsPtr = buffer->getGradientsPtr();

                auto kernelEndTime = std::chrono::high_resolution_clock::now();
                totalKernelCreationUs += std::chrono::duration_cast<std::chrono::microseconds>(kernelEndTime - kernelStartTime).count();
                numKernels++;

                // --- EVALUATION (4 scenarios per execution using SIMD) ---
                auto evalStartTime = std::chrono::high_resolution_clock::now();

                // Pre-allocate working buffers (reused across batches)
                double npvValues[VECTOR_WIDTH];

                // Process paths in batches of 4
                Size numBatches = (config.numPaths + VECTOR_WIDTH - 1) / VECTOR_WIDTH;

                // =================================================================
                // PRE-TRANSPOSE all scenario data for this timestep (ONCE per timestep)
                // =================================================================
                // Layout: transposedInputs[batch * numRF * 4 + rf * 4 + lane]
                // This allows fast _mm256_load_pd instead of slow _mm256_set_pd gather

                // Allocate aligned buffer for AVX2 loads (32-byte alignment)
                size_t transposedSize = numBatches * config.numRiskFactors * VECTOR_WIDTH;
                double* transposedInputs = static_cast<double*>(
#ifdef _WIN32
                    _aligned_malloc(transposedSize * sizeof(double), 32)
#else
                    aligned_alloc(32, transposedSize * sizeof(double))
#endif
                );

                // Transpose: convert from [scenario][riskFactor] to [batch][riskFactor][lane]
                for (Size batch = 0; batch < numBatches; ++batch) {
                    Size batchStart = batch * VECTOR_WIDTH;
                    Size batchSize = std::min(static_cast<Size>(VECTOR_WIDTH), config.numPaths - batchStart);
                    size_t batchBase = batch * config.numRiskFactors * VECTOR_WIDTH;

                    for (Size rf = 0; rf < config.numRiskFactors; ++rf) {
                        size_t rfBase = batchBase + rf * VECTOR_WIDTH;
                        // Copy actual scenario values
                        for (Size lane = 0; lane < batchSize; ++lane) {
                            transposedInputs[rfBase + lane] = scenarios[t][batchStart + lane].flatData[rf];
                        }
                        // Pad remaining lanes with last valid value
                        for (Size lane = batchSize; lane < VECTOR_WIDTH; ++lane) {
                            transposedInputs[rfBase + lane] = transposedInputs[rfBase + batchSize - 1];
                        }
                    }
                }

                for (Size batch = 0; batch < numBatches; ++batch) {
                    Size batchStart = batch * VECTOR_WIDTH;
                    Size batchSize = std::min(static_cast<Size>(VECTOR_WIDTH), config.numPaths - batchStart);

                    // Calculate base offset into transposed buffer for this batch
                    size_t transposedBase = batch * config.numRiskFactors * VECTOR_WIDTH;

                    // Set inputs: DIRECT WRITE with PRE-TRANSPOSED data
                    auto setInputsStart = std::chrono::high_resolution_clock::now();
                    if (allIndicesValid && valuesPtr) {
                        for (Size i = 0; i < config.numRiskFactors; ++i) {
                            __m256d vals = _mm256_load_pd(&transposedInputs[transposedBase + i * VECTOR_WIDTH]);
                            _mm256_store_pd(&valuesPtr[inputIndices[i]], vals);
                        }
                    } else {
                        // Fallback: use setVectorValueDirect
                        double vectorInput[VECTOR_WIDTH];
                        for (Size i = 0; i < config.numRiskFactors; ++i) {
                            vectorInput[0] = transposedInputs[transposedBase + i * VECTOR_WIDTH + 0];
                            vectorInput[1] = transposedInputs[transposedBase + i * VECTOR_WIDTH + 1];
                            vectorInput[2] = transposedInputs[transposedBase + i * VECTOR_WIDTH + 2];
                            vectorInput[3] = transposedInputs[transposedBase + i * VECTOR_WIDTH + 3];
                            buffer->setVectorValueDirect(rateNodeIds[i], vectorInput);
                        }
                    }
                    auto setInputsEnd = std::chrono::high_resolution_clock::now();
                    totalSetInputsUs += std::chrono::duration_cast<std::chrono::nanoseconds>(setInputsEnd - setInputsStart).count() / 1000.0;

                    // Execute kernel - processes all 4 scenarios in parallel
                    auto executeStart = std::chrono::high_resolution_clock::now();
                    kernel->execute(*buffer);
                    auto executeEnd = std::chrono::high_resolution_clock::now();
                    totalExecuteKernelUs += std::chrono::duration_cast<std::chrono::nanoseconds>(executeEnd - executeStart).count() / 1000.0;
                    numEvaluations++;

                    // Get 4 output values
                    auto getOutputsStart = std::chrono::high_resolution_clock::now();
                    if (allIndicesValid && valuesPtr) {
                        npvValues[0] = valuesPtr[outputIndex + 0];
                        npvValues[1] = valuesPtr[outputIndex + 1];
                        npvValues[2] = valuesPtr[outputIndex + 2];
                        npvValues[3] = valuesPtr[outputIndex + 3];
                    } else {
                        buffer->getVectorValueDirect(npvNodeId, npvValues);
                    }
                    auto getOutputsEnd = std::chrono::high_resolution_clock::now();
                    totalGetOutputsUs += std::chrono::duration_cast<std::chrono::nanoseconds>(getOutputsEnd - getOutputsStart).count() / 1000.0;

                    // Get gradients for all 4 lanes
                    auto getGradientsStart = std::chrono::high_resolution_clock::now();
                    double* gradOutputPtrs[4];
                    for (Size b = 0; b < batchSize; ++b) {
                        Size p = batchStart + b;
                        gradOutputPtrs[b] = results.sensitivities[s][t][p].data();
                    }
                    for (Size b = batchSize; b < VECTOR_WIDTH; ++b) {
                        gradOutputPtrs[b] = gradOutputPtrs[batchSize - 1];
                    }

                    if (allIndicesValid) {
                        buffer->getGradientsDirectAllLanes(gradientIndices, gradOutputPtrs);
                    } else {
                        for (Size b = 0; b < batchSize; ++b) {
                            for (Size i = 0; i < config.numRiskFactors; ++i) {
                                std::vector<double> gradVector = buffer->getVectorGradient(rateNodeIds[i]);
                                gradOutputPtrs[b][i] = gradVector[b];
                            }
                        }
                    }
                    auto getGradientsEnd = std::chrono::high_resolution_clock::now();
                    totalGetGradientsUs += std::chrono::duration_cast<std::chrono::nanoseconds>(getGradientsEnd - getGradientsStart).count() / 1000.0;

                    // Handle exposures for each scenario
                    for (Size b = 0; b < batchSize; ++b) {
                        Size p = batchStart + b;
                        results.exposures[s][t][p] = std::max(0.0, npvValues[b]);
                        totalExposure += results.exposures[s][t][p];
                        numScenarios++;
                    }
                }

                // Free the transposed buffer for this timestep
#ifdef _WIN32
                _aligned_free(transposedInputs);
#else
                free(transposedInputs);
#endif

                auto evalEndTime = std::chrono::high_resolution_clock::now();
                totalEvalUs += std::chrono::duration_cast<std::chrono::microseconds>(evalEndTime - evalStartTime).count();
            }
        }

        timing.numKernelsCreated = numKernels;
        timing.numEvaluations = numEvaluations;
        timing.numScenarios = numScenarios;
        timing.kernelCreationTimeMs = totalKernelCreationUs / 1000.0;
        timing.evaluationTimeMs = totalEvalUs / 1000.0;
        timing.setInputsTimeMs = totalSetInputsUs / 1000.0;
        timing.executeKernelTimeMs = totalExecuteKernelUs / 1000.0;
        timing.getOutputsTimeMs = totalGetOutputsUs / 1000.0;
        timing.getGradientsTimeMs = totalGetGradientsUs / 1000.0;
        timing.singleScenarioTimeUs = numScenarios > 0 ? totalEvalUs / numScenarios : 0.0;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalScenarios > 0 ? totalExposure / totalScenarios : 0.0;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
    }

    // Convenience wrappers matching the expected signature for runWithTiming
    // SSE2 variants
    XvaResults computeForgeAadSSE2(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {
        return computeForgeAadSSE2Impl(config, swaps, scenarios, pillars, today, calendar, dayCounter, timing, OptimizationMode::Default);
    }

    XvaResults computeForgeAadSSE2Stability(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {
        return computeForgeAadSSE2Impl(config, swaps, scenarios, pillars, today, calendar, dayCounter, timing, OptimizationMode::StabilityOnly);
    }

    XvaResults computeForgeAadSSE2NoOpt(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {
        return computeForgeAadSSE2Impl(config, swaps, scenarios, pillars, today, calendar, dayCounter, timing, OptimizationMode::NoOpt);
    }

    // AVX2 variants
    XvaResults computeForgeAadAVX2(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {
        return computeForgeAadAVX2Impl(config, swaps, scenarios, pillars, today, calendar, dayCounter, timing, OptimizationMode::Default);
    }

    XvaResults computeForgeAadAVX2Stability(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {
        return computeForgeAadAVX2Impl(config, swaps, scenarios, pillars, today, calendar, dayCounter, timing, OptimizationMode::StabilityOnly);
    }

    XvaResults computeForgeAadAVX2NoOpt(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {
        return computeForgeAadAVX2Impl(config, swaps, scenarios, pillars, today, calendar, dayCounter, timing, OptimizationMode::NoOpt);
    }

    //=========================================================================
    // Run with timing (warmup + timed runs)
    //=========================================================================
    template<typename ComputeFunc>
    TimingResults runWithTiming(
        ComputeFunc computeFunc,
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        XvaResults& lastResults) {

        TimingResults timing;
        TimingResults warmupTiming;

        for (Size w = 0; w < config.warmupRuns; ++w) {
            computeFunc(config, swaps, scenarios, pillars, today, calendar, dayCounter, warmupTiming);
        }

        TimingResults accumulatedTiming;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (Size r = 0; r < config.timedRuns; ++r) {
            TimingResults iterTiming;
            lastResults = computeFunc(config, swaps, scenarios, pillars, today, calendar, dayCounter, iterTiming);
            accumulatedTiming.kernelCreationTimeMs += iterTiming.kernelCreationTimeMs;
            accumulatedTiming.evaluationTimeMs += iterTiming.evaluationTimeMs;
            accumulatedTiming.setInputsTimeMs += iterTiming.setInputsTimeMs;
            accumulatedTiming.executeKernelTimeMs += iterTiming.executeKernelTimeMs;
            accumulatedTiming.getOutputsTimeMs += iterTiming.getOutputsTimeMs;
            accumulatedTiming.getGradientsTimeMs += iterTiming.getGradientsTimeMs;
            accumulatedTiming.numKernelsCreated += iterTiming.numKernelsCreated;
            accumulatedTiming.numEvaluations += iterTiming.numEvaluations;
            accumulatedTiming.numScenarios += iterTiming.numScenarios;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        timing.totalTimeMs = duration.count() / 1000.0;
        timing.avgTimePerIterationMs = timing.totalTimeMs / config.timedRuns;
        timing.kernelCreationTimeMs = accumulatedTiming.kernelCreationTimeMs / config.timedRuns;
        timing.evaluationTimeMs = accumulatedTiming.evaluationTimeMs / config.timedRuns;
        timing.setInputsTimeMs = accumulatedTiming.setInputsTimeMs / config.timedRuns;
        timing.executeKernelTimeMs = accumulatedTiming.executeKernelTimeMs / config.timedRuns;
        timing.getOutputsTimeMs = accumulatedTiming.getOutputsTimeMs / config.timedRuns;
        timing.getGradientsTimeMs = accumulatedTiming.getGradientsTimeMs / config.timedRuns;
        timing.numKernelsCreated = accumulatedTiming.numKernelsCreated / config.timedRuns;
        timing.numEvaluations = accumulatedTiming.numEvaluations / config.timedRuns;
        timing.numScenarios = accumulatedTiming.numScenarios / config.timedRuns;
        timing.singleScenarioTimeUs = accumulatedTiming.numScenarios > 0
            ? (accumulatedTiming.evaluationTimeMs * 1000.0) / accumulatedTiming.numScenarios
            : 0.0;

        return timing;
    }

    //=========================================================================
    // Print results table (7 columns: Bump + 3 SSE2 + 3 AVX2)
    //=========================================================================
    void printResultsTable(
        const std::string& testName,
        const TimingResults& bumpTiming,
        const XvaResults& bumpResults,
        // SSE2 variants
        const TimingResults& sse2Timing,
        const XvaResults& sse2Results,
        const TimingResults& sse2StabTiming,
        const XvaResults& sse2StabResults,
        const TimingResults& sse2NoOptTiming,
        const XvaResults& sse2NoOptResults,
        // AVX2 variants (nullptr if not available)
        const TimingResults* avx2Timing,
        const XvaResults* avx2Results,
        const TimingResults* avx2StabTiming,
        const XvaResults* avx2StabResults,
        const TimingResults* avx2NoOptTiming,
        const XvaResults* avx2NoOptResults,
        const std::vector<IRPillar>& eurPillars,
        const XvaConfig& config) {

        const int col0 = 22;  // Label column
        const int colW = 12;  // Data columns (narrower to fit 7)

        bool hasAvx2 = (avx2Timing != nullptr);

        auto line = [&]() {
            std::cout << "+" << std::string(col0, '-');
            for (int i = 0; i < (hasAvx2 ? 7 : 4); ++i)
                std::cout << "+" << std::string(colW, '-');
            std::cout << "+\n";
        };

        int numCols = hasAvx2 ? 7 : 4;
        int titleWidth = col0 + numCols * (colW + 1) + 1;

        std::cout << "\n";
        line();
        std::cout << "|" << std::setw(titleWidth) << std::left << (" " + testName) << "|\n";
        line();

        std::string configStr = "| Config: " + std::to_string(config.numSwaps) + " swap, "
                              + std::to_string(config.numTimeSteps) + " step" + (config.numTimeSteps > 1 ? "s" : "") + ", "
                              + std::to_string(config.numPaths) + " path" + (config.numPaths > 1 ? "s" : "") + ", "
                              + std::to_string(config.numRiskFactors) + " RF";
        std::cout << std::setw(titleWidth + 1) << std::left << configStr << "|\n";
        line();

        // Header row
        std::cout << "|" << std::setw(col0) << std::left << " Method"
                  << "|" << std::setw(colW) << std::right << "Bump"
                  << "|" << std::setw(colW) << std::right << "SSE2"
                  << "|" << std::setw(colW) << std::right << "SSE2-Stab"
                  << "|" << std::setw(colW) << std::right << "SSE2-NoOpt";
        if (hasAvx2) {
            std::cout << "|" << std::setw(colW) << std::right << "AVX2"
                      << "|" << std::setw(colW) << std::right << "AVX2-Stab"
                      << "|" << std::setw(colW) << std::right << "AVX2-NoOpt";
        }
        std::cout << "|\n";
        line();

        std::cout << std::fixed << std::setprecision(2);

        // Helper lambda to print a row
        auto printRow = [&](const std::string& label,
                           double bump, double s1, double s2, double s3,
                           double a1, double a2, double a3, bool bumpDash = false) {
            std::cout << "|" << std::setw(col0) << std::left << label;
            if (bumpDash) std::cout << "|" << std::setw(colW) << std::right << "-";
            else std::cout << "|" << std::setw(colW) << std::right << bump;
            std::cout << "|" << std::setw(colW) << std::right << s1
                      << "|" << std::setw(colW) << std::right << s2
                      << "|" << std::setw(colW) << std::right << s3;
            if (hasAvx2) {
                std::cout << "|" << std::setw(colW) << std::right << a1
                          << "|" << std::setw(colW) << std::right << a2
                          << "|" << std::setw(colW) << std::right << a3;
            }
            std::cout << "|\n";
        };

        printRow(" Total Time (ms)", bumpTiming.totalTimeMs,
                 sse2Timing.totalTimeMs, sse2StabTiming.totalTimeMs, sse2NoOptTiming.totalTimeMs,
                 hasAvx2 ? avx2Timing->totalTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->totalTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->totalTimeMs : 0);

        printRow(" Avg Time/Iter (ms)", bumpTiming.avgTimePerIterationMs,
                 sse2Timing.avgTimePerIterationMs, sse2StabTiming.avgTimePerIterationMs, sse2NoOptTiming.avgTimePerIterationMs,
                 hasAvx2 ? avx2Timing->avgTimePerIterationMs : 0,
                 hasAvx2 ? avx2StabTiming->avgTimePerIterationMs : 0,
                 hasAvx2 ? avx2NoOptTiming->avgTimePerIterationMs : 0);

        printRow(" Kernel Create (ms)", 0,
                 sse2Timing.kernelCreationTimeMs, sse2StabTiming.kernelCreationTimeMs, sse2NoOptTiming.kernelCreationTimeMs,
                 hasAvx2 ? avx2Timing->kernelCreationTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->kernelCreationTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->kernelCreationTimeMs : 0, true);

        printRow(" Pure Eval (ms)", bumpTiming.evaluationTimeMs,
                 sse2Timing.evaluationTimeMs, sse2StabTiming.evaluationTimeMs, sse2NoOptTiming.evaluationTimeMs,
                 hasAvx2 ? avx2Timing->evaluationTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->evaluationTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->evaluationTimeMs : 0);

        printRow("   - setInputs (ms)", 0,
                 sse2Timing.setInputsTimeMs, sse2StabTiming.setInputsTimeMs, sse2NoOptTiming.setInputsTimeMs,
                 hasAvx2 ? avx2Timing->setInputsTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->setInputsTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->setInputsTimeMs : 0, true);

        printRow("   - execute (ms)", 0,
                 sse2Timing.executeKernelTimeMs, sse2StabTiming.executeKernelTimeMs, sse2NoOptTiming.executeKernelTimeMs,
                 hasAvx2 ? avx2Timing->executeKernelTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->executeKernelTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->executeKernelTimeMs : 0, true);

        printRow("   - getOutputs (ms)", 0,
                 sse2Timing.getOutputsTimeMs, sse2StabTiming.getOutputsTimeMs, sse2NoOptTiming.getOutputsTimeMs,
                 hasAvx2 ? avx2Timing->getOutputsTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->getOutputsTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->getOutputsTimeMs : 0, true);

        printRow("   - getGradients (ms)", 0,
                 sse2Timing.getGradientsTimeMs, sse2StabTiming.getGradientsTimeMs, sse2NoOptTiming.getGradientsTimeMs,
                 hasAvx2 ? avx2Timing->getGradientsTimeMs : 0,
                 hasAvx2 ? avx2StabTiming->getGradientsTimeMs : 0,
                 hasAvx2 ? avx2NoOptTiming->getGradientsTimeMs : 0, true);

        line();

        // Speedup row
        std::cout << "|" << std::setw(col0) << std::left << " Speedup vs Bump";
        std::cout << "|" << std::setw(colW-1) << std::right << "1.00" << "x";
        double spSSE2 = bumpTiming.totalTimeMs / sse2Timing.totalTimeMs;
        double spSSE2Stab = bumpTiming.totalTimeMs / sse2StabTiming.totalTimeMs;
        double spSSE2NoOpt = bumpTiming.totalTimeMs / sse2NoOptTiming.totalTimeMs;
        std::cout << "|" << std::setw(colW-1) << std::right << spSSE2 << "x";
        std::cout << "|" << std::setw(colW-1) << std::right << spSSE2Stab << "x";
        std::cout << "|" << std::setw(colW-1) << std::right << spSSE2NoOpt << "x";
        if (hasAvx2) {
            double spAVX2 = bumpTiming.totalTimeMs / avx2Timing->totalTimeMs;
            double spAVX2Stab = bumpTiming.totalTimeMs / avx2StabTiming->totalTimeMs;
            double spAVX2NoOpt = bumpTiming.totalTimeMs / avx2NoOptTiming->totalTimeMs;
            std::cout << "|" << std::setw(colW-1) << std::right << spAVX2 << "x";
            std::cout << "|" << std::setw(colW-1) << std::right << spAVX2Stab << "x";
            std::cout << "|" << std::setw(colW-1) << std::right << spAVX2NoOpt << "x";
        }
        std::cout << "|\n";
        line();

        // Expected Exposure and CVA
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "|" << std::setw(col0) << std::left << " Expected Exposure"
                  << "|" << std::setw(colW) << std::right << bumpResults.expectedExposure
                  << "|" << std::setw(colW) << std::right << sse2Results.expectedExposure
                  << "|" << std::setw(colW) << std::right << sse2StabResults.expectedExposure
                  << "|" << std::setw(colW) << std::right << sse2NoOptResults.expectedExposure;
        if (hasAvx2) {
            std::cout << "|" << std::setw(colW) << std::right << avx2Results->expectedExposure
                      << "|" << std::setw(colW) << std::right << avx2StabResults->expectedExposure
                      << "|" << std::setw(colW) << std::right << avx2NoOptResults->expectedExposure;
        }
        std::cout << "|\n";

        std::cout << "|" << std::setw(col0) << std::left << " CVA"
                  << "|" << std::setw(colW) << std::right << bumpResults.cva
                  << "|" << std::setw(colW) << std::right << sse2Results.cva
                  << "|" << std::setw(colW) << std::right << sse2StabResults.cva
                  << "|" << std::setw(colW) << std::right << sse2NoOptResults.cva;
        if (hasAvx2) {
            std::cout << "|" << std::setw(colW) << std::right << avx2Results->cva
                      << "|" << std::setw(colW) << std::right << avx2StabResults->cva
                      << "|" << std::setw(colW) << std::right << avx2NoOptResults->cva;
        }
        std::cout << "|\n";
        line();
        std::cout << "\n";
    }

    //=========================================================================
    // Create test case configurations
    //=========================================================================
    std::vector<XvaConfig> createTestCases() {
        std::vector<XvaConfig> configs;

        // Test 1: Simple pricing - 10 risk factors (EUR curve only)
        configs.push_back({"Test 1: Simple (10 RF)", 1, 1, 1, 10, 2, 5, 1e-4});

        // Test 2: Full XVA risk factors - 100 RF, single scenario
        configs.push_back({"Test 2: Full RF (100 RF)", 1, 1, 1, 100, 2, 5, 1e-4});

        // Test 3: Add time steps
        configs.push_back({"Test 3: Time Steps (3 steps)", 1, 3, 1, 100, 2, 5, 1e-4});

        // Test 4: Add MC paths (10)
        configs.push_back({"Test 4: MC 10 paths", 1, 3, 10, 100, 2, 5, 1e-4});

        // Test 5: Add MC paths (100)
        configs.push_back({"Test 5: MC 100 paths", 1, 3, 100, 100, 2, 5, 1e-4});

        // Test 6: Full scale (1000 paths)
        configs.push_back({"Test 6: Full Scale (1000 paths)", 1, 3, 1000, 100, 2, 5, 1e-4});

        return configs;
    }

}  // namespace

//=============================================================================
// MAIN: XVA Forge Benchmark (Bump-Reval vs Forge-SSE2 vs Forge-AVX2)
//       PATCHED VERSION - Requires patched Forge with optimized buffer methods
//=============================================================================
int main(int argc, char* argv[]) {
    std::cout << "=============================================================\n";
    std::cout << "  XVA Benchmark - QuantLib with Forge (SSE2 vs AVX2)\n";
    std::cout << "  PATCHED VERSION - Using optimized buffer access methods\n";
    std::cout << "=============================================================\n";

    // Check AVX2 support
    bool avx2Supported = isAVX2Supported();
    std::cout << "\n  CPU AVX2 Support: " << (avx2Supported ? "YES" : "NO") << "\n";
    if (!avx2Supported) {
        std::cout << "  (AVX2 column will be skipped)\n";
    }
    std::cout << "\n";

    try {
        Calendar calendar = TARGET();
        Date today = Date(15, January, 2024);
        Settings::instance().evaluationDate() = today;
        DayCounter dayCounter = Actual365Fixed();

        auto eurPillars = createEURCurvePillars();
        auto baseRiskFactors = createBaseRiskFactors();
        auto testCases = createTestCases();

        bool allPassed = true;

        for (const auto& config : testCases) {
            std::cout << "\nRunning " << config.name << "...\n";

            auto swaps = createSwapDefinitions(config.numSwaps);
            auto scenarios = generateScenarios(config, eurPillars, baseRiskFactors);

            // 1. Bump-Reval (baseline)
            std::cout << "  Bump-Reval (" << config.numRiskFactors << " RF)...\n";
            XvaResults bumpResults;
            auto bumpTiming = runWithTiming(computeBumpReval, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, bumpResults);

            // 2. SSE2 variants
            std::cout << "  SSE2 (" << config.numRiskFactors << " RF)...\n";
            XvaResults sse2Results;
            auto sse2Timing = runWithTiming(computeForgeAadSSE2, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, sse2Results);

            std::cout << "  SSE2-Stab (" << config.numRiskFactors << " RF)...\n";
            XvaResults sse2StabResults;
            auto sse2StabTiming = runWithTiming(computeForgeAadSSE2Stability, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, sse2StabResults);

            std::cout << "  SSE2-NoOpt (" << config.numRiskFactors << " RF)...\n";
            XvaResults sse2NoOptResults;
            auto sse2NoOptTiming = runWithTiming(computeForgeAadSSE2NoOpt, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, sse2NoOptResults);

            // 3. AVX2 variants (if supported)
            XvaResults avx2Results, avx2StabResults, avx2NoOptResults;
            TimingResults avx2Timing, avx2StabTiming, avx2NoOptTiming;
            if (avx2Supported) {
                std::cout << "  AVX2 (" << config.numRiskFactors << " RF)...\n";
                avx2Timing = runWithTiming(computeForgeAadAVX2, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, avx2Results);

                std::cout << "  AVX2-Stab (" << config.numRiskFactors << " RF)...\n";
                avx2StabTiming = runWithTiming(computeForgeAadAVX2Stability, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, avx2StabResults);

                std::cout << "  AVX2-NoOpt (" << config.numRiskFactors << " RF)...\n";
                avx2NoOptTiming = runWithTiming(computeForgeAadAVX2NoOpt, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, avx2NoOptResults);
            }

            // Print results
            printResultsTable(config.name, bumpTiming, bumpResults,
                              sse2Timing, sse2Results,
                              sse2StabTiming, sse2StabResults,
                              sse2NoOptTiming, sse2NoOptResults,
                              avx2Supported ? &avx2Timing : nullptr,
                              avx2Supported ? &avx2Results : nullptr,
                              avx2Supported ? &avx2StabTiming : nullptr,
                              avx2Supported ? &avx2StabResults : nullptr,
                              avx2Supported ? &avx2NoOptTiming : nullptr,
                              avx2Supported ? &avx2NoOptResults : nullptr,
                              eurPillars, config);
        }

        std::cout << "=============================================================\n";
        if (allPassed) {
            std::cout << "  All benchmarks completed successfully.\n";
        } else {
            std::cout << "  WARNING: Some results did not match within tolerance.\n";
        }
        std::cout << "  Forge AVX benchmark (PATCHED) completed.\n";
        std::cout << "=============================================================\n";

        return allPassed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
