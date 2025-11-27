/*******************************************************************************

   XVA Performance Benchmark - Forge Version (Standalone)

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

******************************************************************************/

// =============================================================================
// XVA PERFORMANCE TEST - FORGE VERSION (Standalone Executable)
// =============================================================================
// This test compares THREE approaches:
//   1. Bump-Reval (baseline)    - Direct QuantLib evaluation with finite differences
//   2. Forge Kernel Bump-Reval  - JIT-compiled kernel, still using bump-reval
//   3. Forge AAD                - JIT-compiled kernel with automatic differentiation
//
// Forge provides two distinct benefits:
//   A) JIT Compilation: Kernel can be re-evaluated with different inputs much faster
//   B) AAD: Gradients computed automatically in single backward pass (no bumping)
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
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace QuantLib;

namespace {

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
        Size numSwaps = 1;
        Size numTimeSteps = 3;
        Size numPaths = 1000;
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
    // Market Scenario (100 risk factors total)
    //=========================================================================
    struct MarketScenario {
        std::vector<double> flatData;

        const std::vector<double>& flatten() const { return flatData; }

        static constexpr Size totalRiskFactors() { return 100; }
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
    // Timing Results
    //=========================================================================
    struct TimingResults {
        double totalTimeMs = 0.0;
        double avgTimePerIterationMs = 0.0;
        double kernelCreationTimeMs = 0.0;
        double evaluationTimeMs = 0.0;
        double singleScenarioTimeUs = 0.0;
        Size numScenarios = 0;
        Size numEvaluations = 0;
        Size numKernelsCreated = 0;
    };

    //=========================================================================
    // Generate scenarios with all 100 risk factors
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
            double timeYears = double(t + 1) / config.numTimeSteps * 5.0;
            double sqrtTime = std::sqrt(timeYears);

            for (Size p = 0; p < config.numPaths; ++p) {
                MarketScenario& sc = scenarios[t][p];
                sc.flatData.resize(100);
                Size idx = 0;

                double eurParallel = dist(gen);
                for (Size i = 0; i < eurPillars.size(); ++i) {
                    sc.flatData[idx++] = eurPillars[i].baseRate + eurPillars[i].volatility * eurParallel * sqrtTime;
                }

                double usdParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseFactors.usdRates[i] + rateVol * usdParallel * sqrtTime;
                }

                double gbpParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseFactors.gbpRates[i] + rateVol * gbpParallel * sqrtTime;
                }

                double jpyParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseFactors.jpyRates[i] + rateVol * jpyParallel * sqrtTime;
                }

                double chfParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseFactors.chfRates[i] + rateVol * chfParallel * sqrtTime;
                }

                for (Size i = 0; i < 5; ++i) {
                    double fxShock = dist(gen);
                    sc.flatData[idx++] = baseFactors.fxRates[i] * std::exp(fxVol * fxShock * sqrtTime - 0.5 * fxVol * fxVol * timeYears);
                }

                double cptyShock = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseFactors.counterpartySpreads[i] * std::exp(creditVol * cptyShock * sqrtTime);
                }

                double ownShock = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseFactors.ownSpreads[i] * std::exp(creditVol * ownShock * sqrtTime);
                }

                double volShock = dist(gen);
                for (Size i = 0; i < 25; ++i) {
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
    // Price swap at time step with all 100 risk factors
    //=========================================================================
    Real priceSwapAtTimeStep(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<Real>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter) {

        double timeStepFraction = double(timeStep) / totalTimeSteps;
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
        const Size numInputs = 100;
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

                    double baseNpv = value(priceSwapAtTimeStep(
                        swaps[s], t, config.numTimeSteps, realInputs, pillars, today, calendar, dayCounter));
                    totalEvaluations++;

                    results.exposures[s][t][p] = std::max(0.0, baseNpv);
                    totalExposure += results.exposures[s][t][p];

                    results.sensitivities[s][t][p].resize(numInputs);
                    for (Size i = 0; i < numInputs; ++i) {
                        std::vector<Real> bumpedInputs = realInputs;
                        bumpedInputs[i] += config.bumpSize;
                        double bumpedNpv = value(priceSwapAtTimeStep(
                            swaps[s], t, config.numTimeSteps, bumpedInputs, pillars, today, calendar, dayCounter));
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
        timing.singleScenarioTimeUs = double(duration.count()) / numScenarios;
        timing.numKernelsCreated = 0;
        timing.kernelCreationTimeMs = 0.0;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalExposure / totalScenarios;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
    }

    //=========================================================================
    // 2. FORGE KERNEL BUMP-REVAL (JIT speedup, still uses bumping)
    //=========================================================================
    XvaResults computeForgeKernelBumpReval(
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
        Size numKernels = 0;
        Size numEvaluations = 0;
        Size numScenarios = 0;
        double totalKernelCreationUs = 0.0;
        double totalEvalUs = 0.0;
        const Size numInputs = 100;

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
                std::vector<Real> rateInputs(numInputs);
                std::vector<forge::NodeId> rateNodeIds(numInputs);
                for (Size i = 0; i < numInputs; ++i) {
                    rateInputs[i] = flatInputs[i];
                    rateInputs[i].markForgeInput();
                    rateNodeIds[i] = rateInputs[i].forgeNodeId();
                }

                Real npv = priceSwapAtTimeStep(swaps[s], t, config.numTimeSteps, rateInputs, pillars, today, calendar, dayCounter);
                npv.markForgeOutput();
                forge::NodeId npvNodeId = npv.forgeNodeId();

                recorder.stop();
                forge::Graph graph = recorder.graph();

                forge::ForgeEngine compiler;
                auto kernel = compiler.compile(graph);
                auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

                auto kernelEndTime = std::chrono::high_resolution_clock::now();
                totalKernelCreationUs += std::chrono::duration_cast<std::chrono::microseconds>(kernelEndTime - kernelStartTime).count();
                numKernels++;

                // --- EVALUATION ---
                auto evalStartTime = std::chrono::high_resolution_clock::now();

                for (Size p = 0; p < config.numPaths; ++p) {
                    const auto& scenario = scenarios[t][p];
                    std::vector<double> scenarioInputs = scenario.flatten();

                    for (Size i = 0; i < numInputs; ++i) {
                        buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                    }
                    kernel->execute(*buffer);
                    numEvaluations++;

                    double baseNpv = buffer->getValue(npvNodeId);
                    results.exposures[s][t][p] = std::max(0.0, baseNpv);
                    totalExposure += results.exposures[s][t][p];

                    results.sensitivities[s][t][p].resize(numInputs);
                    for (Size i = 0; i < numInputs; ++i) {
                        buffer->setValue(rateNodeIds[i], scenarioInputs[i] + config.bumpSize);
                        kernel->execute(*buffer);
                        numEvaluations++;

                        double bumpedNpv = buffer->getValue(npvNodeId);
                        results.sensitivities[s][t][p][i] = (bumpedNpv - baseNpv) / config.bumpSize;

                        buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                    }
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
        timing.singleScenarioTimeUs = totalEvalUs / numScenarios;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalExposure / totalScenarios;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
    }

    //=========================================================================
    // 3. FORGE AAD COMPUTATION (JIT + AAD - no bumping needed)
    //=========================================================================
    XvaResults computeForgeAad(
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
        Size numKernels = 0;
        Size numEvaluations = 0;
        Size numScenarios = 0;
        double totalKernelCreationUs = 0.0;
        double totalEvalUs = 0.0;
        const Size numInputs = 100;

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
                std::vector<Real> rateInputs(numInputs);
                std::vector<forge::NodeId> rateNodeIds(numInputs);
                for (Size i = 0; i < numInputs; ++i) {
                    rateInputs[i] = flatInputs[i];
                    rateInputs[i].markForgeInput();
                    rateNodeIds[i] = rateInputs[i].forgeNodeId();
                }

                Real npv = priceSwapAtTimeStep(swaps[s], t, config.numTimeSteps, rateInputs, pillars, today, calendar, dayCounter);
                npv.markForgeOutput();
                forge::NodeId npvNodeId = npv.forgeNodeId();

                recorder.stop();
                forge::Graph graph = recorder.graph();

                forge::ForgeEngine compiler;
                auto kernel = compiler.compile(graph);
                auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

                // Pre-compute gradient buffer indices
                int vectorWidth = buffer->getVectorWidth();
                std::vector<size_t> gradientIndices(numInputs);
                for (Size i = 0; i < numInputs; ++i) {
                    gradientIndices[i] = static_cast<size_t>(rateNodeIds[i]) * vectorWidth;
                }

                auto kernelEndTime = std::chrono::high_resolution_clock::now();
                totalKernelCreationUs += std::chrono::duration_cast<std::chrono::microseconds>(kernelEndTime - kernelStartTime).count();
                numKernels++;

                // --- EVALUATION ---
                auto evalStartTime = std::chrono::high_resolution_clock::now();

                std::vector<double> gradOutput(numInputs);

                for (Size p = 0; p < config.numPaths; ++p) {
                    const auto& scenario = scenarios[t][p];
                    std::vector<double> scenarioInputs = scenario.flatten();

                    for (Size i = 0; i < numInputs; ++i) {
                        buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                    }

                    kernel->execute(*buffer);
                    numEvaluations++;

                    double npvValue = buffer->getValue(npvNodeId);
                    results.exposures[s][t][p] = std::max(0.0, npvValue);
                    totalExposure += results.exposures[s][t][p];

                    buffer->getGradientsDirect(gradientIndices, gradOutput.data());
                    results.sensitivities[s][t][p] = gradOutput;
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
        timing.singleScenarioTimeUs = totalEvalUs / numScenarios;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalExposure / totalScenarios;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
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
        timing.numKernelsCreated = accumulatedTiming.numKernelsCreated / config.timedRuns;
        timing.numEvaluations = accumulatedTiming.numEvaluations / config.timedRuns;
        timing.numScenarios = accumulatedTiming.numScenarios / config.timedRuns;
        timing.singleScenarioTimeUs = (accumulatedTiming.evaluationTimeMs * 1000.0) / accumulatedTiming.numScenarios;

        return timing;
    }

    //=========================================================================
    // Print results table (3 columns)
    //=========================================================================
    void printResultsTable(
        const TimingResults& bumpTiming,
        const XvaResults& bumpResults,
        const TimingResults& forgeBumpTiming,
        const XvaResults& forgeBumpResults,
        const TimingResults& forgeAadTiming,
        const XvaResults& forgeAadResults,
        const std::vector<IRPillar>& eurPillars,
        const XvaConfig& config) {

        const int col1 = 24;
        const int col2 = 16;
        const int col3 = 16;
        const int col4 = 16;
        const Size numInputs = 100;

        auto line = [&]() {
            std::cout << "+" << std::string(col1, '-') << "+"
                      << std::string(col2, '-') << "+"
                      << std::string(col3, '-') << "+"
                      << std::string(col4, '-') << "+\n";
        };

        std::cout << "\n";
        line();
        std::cout << "|                   XVA PERFORMANCE COMPARISON (FORGE)                     |\n";
        line();
        std::cout << "| Config: " << config.numSwaps << " swaps, "
                  << config.numTimeSteps << " steps, "
                  << config.numPaths << " paths, "
                  << numInputs << " risk factors"
                  << std::string(8, ' ') << "|\n";
        line();

        std::cout << "|" << std::setw(col1) << std::left << " Method"
                  << "|" << std::setw(col2) << std::right << "Bump-Reval"
                  << "|" << std::setw(col3) << std::right << "Forge-Bump"
                  << "|" << std::setw(col4) << std::right << "Forge-AAD" << "|\n";
        line();

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "|" << std::setw(col1) << std::left << " Total Time (ms)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.totalTimeMs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.totalTimeMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.totalTimeMs << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Avg Time/Iter (ms)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.avgTimePerIterationMs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.avgTimePerIterationMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.avgTimePerIterationMs << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Kernel Create (ms)"
                  << "|" << std::setw(col2) << std::right << "-"
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.kernelCreationTimeMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.kernelCreationTimeMs << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Pure Eval (ms)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.evaluationTimeMs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.evaluationTimeMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.evaluationTimeMs << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " # Scenarios"
                  << "|" << std::setw(col2) << std::right << bumpTiming.numScenarios
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.numScenarios
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.numScenarios << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " # Evaluations"
                  << "|" << std::setw(col2) << std::right << bumpTiming.numEvaluations
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.numEvaluations
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.numEvaluations << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Time/Scenario (us)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.singleScenarioTimeUs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.singleScenarioTimeUs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.singleScenarioTimeUs << "|\n";
        line();

        double speedupForgeBump = bumpTiming.totalTimeMs / forgeBumpTiming.totalTimeMs;
        double speedupForgeAad = bumpTiming.totalTimeMs / forgeAadTiming.totalTimeMs;
        std::cout << "|" << std::setw(col1) << std::left << " Speedup"
                  << "|" << std::setw(col2-1) << std::right << "1.00" << "x"
                  << "|" << std::setw(col3-1) << std::right << speedupForgeBump << "x"
                  << "|" << std::setw(col4-1) << std::right << speedupForgeAad << "x|\n";

        Size evalsPerScenarioBump = 1 + numInputs;
        std::cout << "|" << std::setw(col1) << std::left << " Evals/Scenario"
                  << "|" << std::setw(col2) << std::right << evalsPerScenarioBump
                  << "|" << std::setw(col3) << std::right << evalsPerScenarioBump
                  << "|" << std::setw(col4) << std::right << "1" << "|\n";
        std::cout << "|" << std::setw(col1) << std::left << " Expected AAD Benefit"
                  << "|" << std::setw(col2) << std::right << "-"
                  << "|" << std::setw(col3) << std::right << "-"
                  << "|" << std::setw(col4-1) << std::right << evalsPerScenarioBump << "x|\n";
        line();

        std::cout << std::scientific << std::setprecision(4);
        std::cout << "|" << std::setw(col1) << std::left << " Expected Exposure"
                  << "|" << std::setw(col2) << std::right << bumpResults.expectedExposure
                  << "|" << std::setw(col3) << std::right << forgeBumpResults.expectedExposure
                  << "|" << std::setw(col4) << std::right << forgeAadResults.expectedExposure << "|\n";
        std::cout << "|" << std::setw(col1) << std::left << " CVA"
                  << "|" << std::setw(col2) << std::right << bumpResults.cva
                  << "|" << std::setw(col3) << std::right << forgeBumpResults.cva
                  << "|" << std::setw(col4) << std::right << forgeAadResults.cva << "|\n";
        line();

        std::cout << "| Sample Sensitivities (Swap 0, Time 0, Path 0):" << std::string(25, ' ') << "|\n";
        line();

        for (Size i = 0; i < std::min(eurPillars.size(), Size(5)); ++i) {
            std::string label = " dNPV/d(" + eurPillars[i].name + ")";
            std::cout << "|" << std::setw(col1) << std::left << label
                      << "|" << std::setw(col2) << std::right << bumpResults.sensitivities[0][0][0][i]
                      << "|" << std::setw(col3) << std::right << forgeBumpResults.sensitivities[0][0][0][i]
                      << "|" << std::setw(col4) << std::right << forgeAadResults.sensitivities[0][0][0][i] << "|\n";
        }
        std::cout << "|" << std::setw(col1) << std::left << " ... (95 more)"
                  << "|" << std::setw(col2) << " "
                  << "|" << std::setw(col3) << " "
                  << "|" << std::setw(col4) << " " << "|\n";
        line();
        std::cout << "\n";
    }

    //=========================================================================
    // Verify results match
    //=========================================================================
    bool verifyResults(
        const XvaResults& bumpResults,
        const XvaResults& forgeBumpResults,
        const XvaResults& forgeAadResults,
        double tolerance = 0.001) {

        bool passed = true;
        const Size numInputs = 100;

        // Check expected exposure
        double exposureDiff1 = std::abs(bumpResults.expectedExposure - forgeBumpResults.expectedExposure);
        double exposureDiff2 = std::abs(bumpResults.expectedExposure - forgeAadResults.expectedExposure);
        double exposureTol = std::abs(bumpResults.expectedExposure) * tolerance;

        if (exposureDiff1 > exposureTol) {
            std::cerr << "WARNING: Exposure mismatch Bump vs Forge-Bump: " << exposureDiff1 << " > " << exposureTol << "\n";
            passed = false;
        }
        if (exposureDiff2 > exposureTol) {
            std::cerr << "WARNING: Exposure mismatch Bump vs Forge-AAD: " << exposureDiff2 << " > " << exposureTol << "\n";
            passed = false;
        }

        // Check sensitivities for first scenario
        for (Size i = 0; i < numInputs; ++i) {
            double bumpSens = bumpResults.sensitivities[0][0][0][i];
            if (std::abs(bumpSens) > 1e-10) {
                double forgeBumpSens = forgeBumpResults.sensitivities[0][0][0][i];
                double forgeAadSens = forgeAadResults.sensitivities[0][0][0][i];

                double diff1 = std::abs(bumpSens - forgeBumpSens) / std::abs(bumpSens);
                double diff2 = std::abs(bumpSens - forgeAadSens) / std::abs(bumpSens);

                if (diff1 > 0.01) {  // 1% tolerance for sensitivities
                    std::cerr << "WARNING: Sensitivity[" << i << "] mismatch Bump vs Forge-Bump: "
                              << bumpSens << " vs " << forgeBumpSens << " (" << diff1*100 << "%)\n";
                    passed = false;
                }
                if (diff2 > 0.01) {
                    std::cerr << "WARNING: Sensitivity[" << i << "] mismatch Bump vs Forge-AAD: "
                              << bumpSens << " vs " << forgeAadSens << " (" << diff2*100 << "%)\n";
                    passed = false;
                }
            }
        }

        return passed;
    }

}  // namespace

//=============================================================================
// MAIN: XVA Forge Benchmark (Bump-Reval vs Forge-Bump vs Forge-AAD)
//=============================================================================
int main(int argc, char* argv[]) {
    std::cout << "=============================================================\n";
    std::cout << "  XVA Benchmark - QuantLib with Forge (JIT + AAD)\n";
    std::cout << "=============================================================\n";

    try {
        XvaConfig config;
        config.numSwaps = 1;
        config.numTimeSteps = 3;
        config.numPaths = 1000;
        config.warmupRuns = 2;
        config.timedRuns = 5;

        Calendar calendar = TARGET();
        Date today = Date(15, January, 2024);
        Settings::instance().evaluationDate() = today;
        DayCounter dayCounter = Actual365Fixed();

        auto eurPillars = createEURCurvePillars();
        auto baseRiskFactors = createBaseRiskFactors();
        auto swaps = createSwapDefinitions(config.numSwaps);
        auto scenarios = generateScenarios(config, eurPillars, baseRiskFactors);

        // 1. Bump-Reval (baseline)
        std::cout << "\nRunning Bump-Reval baseline (100 risk factors)...\n";
        XvaResults bumpResults;
        auto bumpTiming = runWithTiming(computeBumpReval, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, bumpResults);

        // 2. Forge Kernel Bump-Reval (JIT speedup only)
        std::cout << "Running Forge-Bump (100 risk factors)...\n";
        XvaResults forgeBumpResults;
        auto forgeBumpTiming = runWithTiming(computeForgeKernelBumpReval, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, forgeBumpResults);

        // 3. Forge AAD (JIT + AAD)
        std::cout << "Running Forge-AAD (100 risk factors)...\n";
        XvaResults forgeAadResults;
        auto forgeAadTiming = runWithTiming(computeForgeAad, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, forgeAadResults);

        // Print results
        printResultsTable(bumpTiming, bumpResults,
                          forgeBumpTiming, forgeBumpResults,
                          forgeAadTiming, forgeAadResults,
                          eurPillars, config);

        // Verify results match
        bool verified = verifyResults(bumpResults, forgeBumpResults, forgeAadResults);

        if (verified) {
            std::cout << "All results verified successfully.\n";
        } else {
            std::cout << "WARNING: Some results did not match within tolerance.\n";
        }

        std::cout << "Forge benchmark completed successfully.\n";
        return verified ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
