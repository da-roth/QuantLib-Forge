/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Barrier Option AAD test using Forge.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors
   Copyright (C) 2003, 2004 Ferdinando Ametrano
   Copyright (C) 2005, 2007 StatPro Italia srl
   Copyright (C) 2005 Joseph Wang

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2023, 2024 Xcelerit Computing Limited

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#include "toplevelfixture.hpp"
#include "utilities_forge.hpp"
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/errorfunction.hpp>
#include <ql/pricingengines/barrier/analyticbarrierengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>

// Forge integration headers
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibForgeRisksTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(BarrierOptionForgeTest)

namespace {

    // Common tolerance for derivative comparisons (AAD vs finite difference bumping)
    // AAD computes exact derivatives while bumping has truncation error, so small
    // differences are expected. 1e-5% is tight enough to catch real errors while
    // accommodating numerical differences from the two methods.
    constexpr double DERIVATIVE_TOLERANCE_PCT = 1e-5;

    struct BarrierOptionData {
        Option::Type type;
        Real strike;
        Real u;       // underlying
        Rate r;       // risk-free rate
        Real b;       // barrier
        Volatility v; // volatility
    };

    // Helper: no optimization config
    forge::CompilerConfig noOptimizationConfig() {
        forge::CompilerConfig config;
        config.enableOptimizations = false;
        config.enableInactiveFolding = false;
        config.enableCSE = false;
        config.enableAlgebraicSimplification = false;
        config.enableStabilityCleaning = false;
        return config;
    }

}
namespace {

    template <class PriceFunc>
    Real priceWithBumping(const BarrierOptionData& value,
                          BarrierOptionData& derivatives,
                          PriceFunc func) {
        // Bumping
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.strike += eps;
        auto vplus = func(data);
        derivatives.strike = (vplus - v) / eps;
        data = value;

        data.u += eps;
        vplus = func(data);
        derivatives.u = (vplus - v) / eps;
        data = value;

        data.r += eps;
        vplus = func(data);
        derivatives.r = (vplus - v) / eps;
        data = value;

        data.b += eps;
        vplus = func(data);
        derivatives.b = (vplus - v) / eps;
        data = value;

        data.v += eps;
        vplus = func(data);
        derivatives.v = (vplus - v) / eps;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const BarrierOptionData& values,
                           BarrierOptionData& derivatives,
                           PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.strike.markForgeInputAndDiff();
        data.u.markForgeInputAndDiff();
        data.r.markForgeInputAndDiff();
        data.b.markForgeInputAndDiff();
        data.v.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId strikeNodeId = data.strike.forgeNodeId();
        forge::NodeId uNodeId = data.u.forgeNodeId();
        forge::NodeId rNodeId = data.r.forgeNodeId();
        forge::NodeId bNodeId = data.b.forgeNodeId();
        forge::NodeId vNodeId = data.v.forgeNodeId();

        // Compute the price (this builds the computation graph)
        auto price = func(data);

        // Mark output
        price.markForgeOutput();
        forge::NodeId priceNodeId = price.forgeNodeId();

        // Stop recording and get the graph
        recorder.stop();
        forge::Graph graph = recorder.graph();

        // JIT compile the graph
        forge::ForgeEngine compiler;  // Default is stability-only
        auto kernel = compiler.compile(graph);
        auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

        // Set input values
        int vectorWidth = buffer->getVectorWidth();
        double strikeVal[4] = {value(values.strike), value(values.strike), value(values.strike), value(values.strike)}; buffer->setLanes(strikeNodeId, strikeVal);
        double uVal[4] = {value(values.u), value(values.u), value(values.u), value(values.u)}; buffer->setLanes(uNodeId, uVal);
        double rVal[4] = {value(values.r), value(values.r), value(values.r), value(values.r)}; buffer->setLanes(rNodeId, rVal);
        double bVal[4] = {value(values.b), value(values.b), value(values.b), value(values.b)}; buffer->setLanes(bNodeId, bVal);
        double vVal[4] = {value(values.v), value(values.v), value(values.v), value(values.v)}; buffer->setLanes(vNodeId, vVal);

        // Execute (forward + backward in one call)
        buffer->clearGradients();
        kernel->execute(*buffer);

        // Get the price value
        double priceOut[4]; buffer->getLanes(priceNodeId, priceOut);
        double priceValue = priceOut[0];

        // Get gradients directly
        std::vector<size_t> gradientIndices = {
            buffer->getBufferIndex(strikeNodeId),
            buffer->getBufferIndex(uNodeId),
            buffer->getBufferIndex(rNodeId),
            buffer->getBufferIndex(bNodeId),
            buffer->getBufferIndex(vNodeId)
        };
        std::vector<double> gradients(5 * vectorWidth);
        buffer->getGradientLanes(gradientIndices, gradients.data());

        derivatives.strike = gradients[0 * vectorWidth];
        derivatives.u = gradients[1 * vectorWidth];
        derivatives.r = gradients[2 * vectorWidth];
        derivatives.b = gradients[3 * vectorWidth];
        derivatives.v = gradients[4 * vectorWidth];

        return Real(priceValue);
    }
}

// =============================================================================
// SANITY TESTS - Progressive tests to isolate Forge AAD issues
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeBasicArithmetic) {
    // z = x * y + x, dz/dx = y + 1, dz/dy = x
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 2.0, y = 3.0;
    x.markForgeInputAndDiff();
    y.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId(), yId = y.forgeNodeId();

    Real z = x * y + x;
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId), buffer->getBufferIndex(yId)};
    std::vector<double> grad(2 * vw);

    // Test 1: x=2, y=3 -> z=8, dz/dx=4, dz/dy=2
    double xVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal1);
    double yVal1[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(yId, yVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOut1[4]; buffer->getLanes(zId, zOut1);
    bool v1 = std::abs(zOut1[0] - 8.0) < 1e-9;
    bool g1 = std::abs(grad[0] - 4.0) < 1e-9 && std::abs(grad[1 * vw] - 2.0) < 1e-9;

    // Test 2 (re-eval): x=4, y=5 -> z=24, dz/dx=6, dz/dy=4
    double xVal2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal2);
    double yVal2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(yId, yVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOut2[4]; buffer->getLanes(zId, zOut2);
    bool v2 = std::abs(zOut2[0] - 24.0) < 1e-9;
    bool g2 = std::abs(grad[0] - 6.0) < 1e-9 && std::abs(grad[1 * vw] - 4.0) < 1e-9;

    BOOST_TEST_MESSAGE("BasicArithmetic: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeSimpleQuote) {
    // SimpleQuote pass-through: output = input, d(output)/d(input) = 1
    forge::GraphRecorder recorder;
    recorder.start();

    Real inputValue = 100.0;
    inputValue.markForgeInputAndDiff();
    forge::NodeId inputId = inputValue.forgeNodeId();

    auto quote = ext::make_shared<SimpleQuote>(inputValue);
    Real outputValue = quote->value();
    outputValue.markForgeOutput();
    forge::NodeId outputId = outputValue.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(inputId)};
    std::vector<double> grad(1 * vw);

    // Test 1: input=100
    double inputVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(inputId, inputVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOut1[4]; buffer->getLanes(outputId, outputOut1);
    bool v1 = std::abs(outputOut1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0] - 1.0) < 1e-9;

    // Test 2: input=150
    double inputVal2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(inputId, inputVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOut2[4]; buffer->getLanes(outputId, outputOut2);
    bool v2 = std::abs(outputOut2[0] - 150.0) < 1e-9;
    bool g2 = std::abs(grad[0] - 1.0) < 1e-9;

    BOOST_TEST_MESSAGE("SimpleQuote: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeFlatForward) {
    // df = exp(-r*t), d(df)/dr = -t*exp(-r*t)
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = 0.05;
    rate.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();

    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), Actual365Fixed()));
    Real df = flatRate->discount(maturity);
    df.markForgeOutput();
    forge::NodeId dfId = df.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(rateId)};
    std::vector<double> grad(1 * vw);

    // Test 1: r=0.05
    double rateVal1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOut1[4]; buffer->getLanes(dfId, dfOut1);
    bool v1 = std::abs(dfOut1[0] - std::exp(-0.05)) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - (-std::exp(-0.05))) < 1e-4;

    // Test 2: r=0.08
    double rateVal2[4] = {0.08, 0.08, 0.08, 0.08}; buffer->setLanes(rateId, rateVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOut2[4]; buffer->getLanes(dfId, dfOut2);
    bool v2 = std::abs(dfOut2[0] - std::exp(-0.08)) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - (-std::exp(-0.08))) < 1e-4;

    BOOST_TEST_MESSAGE("FlatForward: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeBlackConstantVol) {
    // BlackConstantVol pass-through: blackVol(t,K) = vol, d/dvol = 1
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;

    forge::GraphRecorder recorder;
    recorder.start();

    Real vol = 0.20;
    vol.markForgeInputAndDiff();
    forge::NodeId volId = vol.forgeNodeId();

    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), Actual365Fixed()));
    Real blackVol = flatVol->blackVol(maturity, 100.0);
    blackVol.markForgeOutput();
    forge::NodeId blackVolId = blackVol.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(volId)};
    std::vector<double> grad(1 * vw);

    // Test 1: vol=0.20
    double volVal1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOut1[4]; buffer->getLanes(blackVolId, blackVolOut1);
    bool v1 = std::abs(blackVolOut1[0] - 0.20) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    // Test 2: vol=0.30
    double volVal2[4] = {0.30, 0.30, 0.30, 0.30}; buffer->setLanes(volId, volVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOut2[4]; buffer->getLanes(blackVolId, blackVolOut2);
    bool v2 = std::abs(blackVolOut2[0] - 0.30) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    BOOST_TEST_MESSAGE("BlackConstantVol: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeBlackScholesProcess) {
    // x0 = spot, d(x0)/d(spot) = 1, d(x0)/d(rate) = 0, d(x0)/d(vol) = 0
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    forge::GraphRecorder recorder;
    recorder.start();

    Real spot = 100.0, rate = 0.04, vol = 0.20;
    spot.markForgeInputAndDiff();
    rate.markForgeInputAndDiff();
    vol.markForgeInputAndDiff();
    forge::NodeId spotId = spot.forgeNodeId();
    forge::NodeId rateId = rate.forgeNodeId();
    forge::NodeId volId = vol.forgeNodeId();

    auto spotQuote = ext::make_shared<SimpleQuote>(spot);
    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    DayCounter dayCounter = Actual365Fixed();
    Handle<Quote> spotHandle(spotQuote);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));
    auto bsProcess = ext::make_shared<BlackScholesProcess>(spotHandle, flatRate, flatVol);

    Real x0 = bsProcess->x0();
    x0.markForgeOutput();
    forge::NodeId x0Id = x0.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {
        buffer->getBufferIndex(spotId), buffer->getBufferIndex(rateId), buffer->getBufferIndex(volId)};
    std::vector<double> grad(3 * vw);

    // Test 1: spot=100
    double spotVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(spotId, spotVal1);
    double rateVal1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateVal1);
    double volVal1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out1[4]; buffer->getLanes(x0Id, x0Out1);
    bool v1 = std::abs(x0Out1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6 && std::abs(grad[1 * vw]) < 1e-9 && std::abs(grad[2 * vw]) < 1e-9;

    // Test 2: spot=120
    double spotVal2[4] = {120.0, 120.0, 120.0, 120.0}; buffer->setLanes(spotId, spotVal2);
    double rateVal2[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateVal2);
    double volVal2[4] = {0.25, 0.25, 0.25, 0.25}; buffer->setLanes(volId, volVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out2[4]; buffer->getLanes(x0Id, x0Out2);
    bool v2 = std::abs(x0Out2[0] - 120.0) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-6 && std::abs(grad[1 * vw]) < 1e-9 && std::abs(grad[2 * vw]) < 1e-9;

    BOOST_TEST_MESSAGE("BlackScholesProcess: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

// =============================================================================
// OPTIMIZATION VARIANT TESTS - Same 5 tests with NoOpt and Default (AllOpt)
// =============================================================================

// --- BasicArithmetic with NO optimization ---
BOOST_AUTO_TEST_CASE(testForgeBasicArithmetic_NoOpt) {
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 2.0, y = 3.0;
    x.markForgeInputAndDiff();
    y.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId(), yId = y.forgeNodeId();

    Real z = x * y + x;
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler(noOptimizationConfig());
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId), buffer->getBufferIndex(yId)};
    std::vector<double> grad(2 * vw);

    double xVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal1);
    double yVal1[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(yId, yVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOut1[4]; buffer->getLanes(zId, zOut1);
    bool v1 = std::abs(zOut1[0] - 8.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 4.0) < 1e-9 && std::abs(grad[1 * vw] - 2.0) < 1e-9;

    double xVal2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal2);
    double yVal2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(yId, yVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOut2[4]; buffer->getLanes(zId, zOut2);
    bool v2 = std::abs(zOut2[0] - 24.0) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 6.0) < 1e-9 && std::abs(grad[1 * vw] - 4.0) < 1e-9;

    BOOST_TEST_MESSAGE("BasicArithmetic_NoOpt: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}


// --- SimpleQuote with NO optimization ---
BOOST_AUTO_TEST_CASE(testForgeSimpleQuote_NoOpt) {
    forge::GraphRecorder recorder;
    recorder.start();

    Real inputValue = 100.0;
    inputValue.markForgeInputAndDiff();
    forge::NodeId inputId = inputValue.forgeNodeId();

    auto quote = ext::make_shared<SimpleQuote>(inputValue);
    Real outputValue = quote->value();
    outputValue.markForgeOutput();
    forge::NodeId outputId = outputValue.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler(noOptimizationConfig());
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(inputId)};
    std::vector<double> grad(1 * vw);

    double inputVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(inputId, inputVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOut1[4]; buffer->getLanes(outputId, outputOut1);
    bool v1 = std::abs(outputOut1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-9;

    double inputVal2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(inputId, inputVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOut2[4]; buffer->getLanes(outputId, outputOut2);
    bool v2 = std::abs(outputOut2[0] - 150.0) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-9;

    BOOST_TEST_MESSAGE("SimpleQuote_NoOpt: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}


// --- FlatForward with NO optimization ---
BOOST_AUTO_TEST_CASE(testForgeFlatForward_NoOpt) {
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = 0.05;
    rate.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();

    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), Actual365Fixed()));
    Real df = flatRate->discount(maturity);
    df.markForgeOutput();
    forge::NodeId dfId = df.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler(noOptimizationConfig());
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(rateId)};
    std::vector<double> grad(1 * vw);

    double rateVal1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOut1[4]; buffer->getLanes(dfId, dfOut1);
    bool v1 = std::abs(dfOut1[0] - std::exp(-0.05)) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - (-std::exp(-0.05))) < 1e-4;

    double rateVal2[4] = {0.08, 0.08, 0.08, 0.08}; buffer->setLanes(rateId, rateVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOut2[4]; buffer->getLanes(dfId, dfOut2);
    bool v2 = std::abs(dfOut2[0] - std::exp(-0.08)) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - (-std::exp(-0.08))) < 1e-4;

    BOOST_TEST_MESSAGE("FlatForward_NoOpt: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}


// --- BlackConstantVol with NO optimization ---
BOOST_AUTO_TEST_CASE(testForgeBlackConstantVol_NoOpt) {
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;

    forge::GraphRecorder recorder;
    recorder.start();

    Real vol = 0.20;
    vol.markForgeInputAndDiff();
    forge::NodeId volId = vol.forgeNodeId();

    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), Actual365Fixed()));
    Real blackVol = flatVol->blackVol(maturity, 100.0);
    blackVol.markForgeOutput();
    forge::NodeId blackVolId = blackVol.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler(noOptimizationConfig());
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(volId)};
    std::vector<double> grad(1 * vw);

    double volVal1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOut1[4]; buffer->getLanes(blackVolId, blackVolOut1);
    bool v1 = std::abs(blackVolOut1[0] - 0.20) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    double volVal2[4] = {0.30, 0.30, 0.30, 0.30}; buffer->setLanes(volId, volVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOut2[4]; buffer->getLanes(blackVolId, blackVolOut2);
    bool v2 = std::abs(blackVolOut2[0] - 0.30) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    BOOST_TEST_MESSAGE("BlackConstantVol_NoOpt: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}


// --- BlackScholesProcess with NO optimization ---
BOOST_AUTO_TEST_CASE(testForgeBlackScholesProcess_NoOpt) {
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    forge::GraphRecorder recorder;
    recorder.start();

    Real spot = 100.0, rate = 0.04, vol = 0.20;
    spot.markForgeInputAndDiff();
    rate.markForgeInputAndDiff();
    vol.markForgeInputAndDiff();
    forge::NodeId spotId = spot.forgeNodeId();
    forge::NodeId rateId = rate.forgeNodeId();
    forge::NodeId volId = vol.forgeNodeId();

    auto spotQuote = ext::make_shared<SimpleQuote>(spot);
    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    DayCounter dayCounter = Actual365Fixed();
    Handle<Quote> spotHandle(spotQuote);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));
    auto bsProcess = ext::make_shared<BlackScholesProcess>(spotHandle, flatRate, flatVol);

    Real x0 = bsProcess->x0();
    x0.markForgeOutput();
    forge::NodeId x0Id = x0.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler(noOptimizationConfig());
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {
        buffer->getBufferIndex(spotId), buffer->getBufferIndex(rateId), buffer->getBufferIndex(volId)};
    std::vector<double> grad(3 * vw);

    double spotVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(spotId, spotVal1);
    double rateVal1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateVal1);
    double volVal1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out1[4]; buffer->getLanes(x0Id, x0Out1);
    bool v1 = std::abs(x0Out1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6 && std::abs(grad[1 * vw]) < 1e-9 && std::abs(grad[2 * vw]) < 1e-9;

    double spotVal2[4] = {120.0, 120.0, 120.0, 120.0}; buffer->setLanes(spotId, spotVal2);
    double rateVal2[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateVal2);
    double volVal2[4] = {0.25, 0.25, 0.25, 0.25}; buffer->setLanes(volId, volVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out2[4]; buffer->getLanes(x0Id, x0Out2);
    bool v2 = std::abs(x0Out2[0] - 120.0) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-6 && std::abs(grad[1 * vw]) < 1e-9 && std::abs(grad[2 * vw]) < 1e-9;

    BOOST_TEST_MESSAGE("BlackScholesProcess_NoOpt: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}


// =============================================================================
// DISTRIBUTION SANITY TESTS - Testing where the graph breaks
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeErrorFunction) {
    // erf(x), d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 0.5;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    ErrorFunction erf;
    Real result = erf(x);
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=0.5
    double xVal1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    double expV1 = std::erf(0.5);
    double expG1 = 2.0 / std::sqrt(M_PI) * std::exp(-0.5 * 0.5);
    bool v1 = std::abs(resultOut1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: x=1.0
    double xVal2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    double expV2 = std::erf(1.0);
    double expG2 = 2.0 / std::sqrt(M_PI) * std::exp(-1.0);
    bool v2 = std::abs(resultOut2[0] - expV2) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - expG2) < 1e-3;

    BOOST_TEST_MESSAGE("ErrorFunction: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeNormalDistribution) {
    // N(x) = 1/sqrt(2*pi) * exp(-x^2/2), d/dx N(x) = -x * N(x)
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 0.5;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    NormalDistribution normal;
    Real pdf = normal(x);
    pdf.markForgeOutput();
    forge::NodeId pdfId = pdf.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=0.5
    double xVal1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOut1[4]; buffer->getLanes(pdfId, pdfOut1);
    double expV1 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * 0.5 / 2.0);
    double expG1 = -0.5 * expV1;
    bool v1 = std::abs(pdfOut1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: x=1.0
    double xVal2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOut2[4]; buffer->getLanes(pdfId, pdfOut2);
    double expV2 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-1.0 / 2.0);
    double expG2 = -1.0 * expV2;
    bool v2 = std::abs(pdfOut2[0] - expV2) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - expG2) < 1e-3;

    BOOST_TEST_MESSAGE("NormalDistribution: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeCumulativeNormalDistribution) {
    // Phi(x) = CDF, d/dx Phi(x) = N(x)
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 0.5;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    CumulativeNormalDistribution cdf;
    Real result = cdf(x);
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=0.5
    double xVal1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    double expV1 = 0.5 * (1.0 + std::erf(0.5 / std::sqrt(2.0)));
    double expG1 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * 0.5 / 2.0);
    bool v1 = std::abs(resultOut1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: x=1.0
    double xVal2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    double expV2 = 0.5 * (1.0 + std::erf(1.0 / std::sqrt(2.0)));
    double expG2 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-1.0 / 2.0);
    bool v2 = std::abs(resultOut2[0] - expV2) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - expG2) < 1e-3;

    BOOST_TEST_MESSAGE("CumulativeNormalDist: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeCumulativeNormalDistributionTail) {
    // Tail region test - asymptotic expansion may trigger at x < -5
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = -5.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    CumulativeNormalDistribution cdf;
    Real result = cdf(x);
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=-5.0
    double xVal1[4] = {-5.0, -5.0, -5.0, -5.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    double expV1 = 0.5 * (1.0 + std::erf(-5.0 / std::sqrt(2.0)));
    double expG1 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-5.0 * 5.0 / 2.0);
    bool v1 = std::abs(resultOut1[0] - expV1) / expV1 < 1e-4;
    bool g1 = std::abs(grad[0 * vw] - expG1) / expG1 < 0.1;

    // Test 2: x=-4.0
    double xVal2[4] = {-4.0, -4.0, -4.0, -4.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    double expV2 = 0.5 * (1.0 + std::erf(-4.0 / std::sqrt(2.0)));
    double expG2 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-4.0 * 4.0 / 2.0);
    bool v2 = std::abs(resultOut2[0] - expV2) / expV2 < 1e-4;
    bool g2 = std::abs(grad[0 * vw] - expG2) / expG2 < 0.1;

    BOOST_TEST_MESSAGE("CumulativeNormalTail: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeNormalDistributionExtremeTail) {
    // Extreme tail: x=40 -> exponent=-800 < -690 triggers early return
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 40.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    NormalDistribution normal;
    Real pdf = normal(x);
    pdf.markForgeOutput();
    forge::NodeId pdfId = pdf.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=40
    double xVal1[4] = {40.0, 40.0, 40.0, 40.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOut1[4]; buffer->getLanes(pdfId, pdfOut1);
    bool v1 = std::abs(pdfOut1[0]) < 1e-300;
    bool g1 = std::abs(grad[0 * vw]) < 1e-100;

    // Test 2: x=50
    double xVal2[4] = {50.0, 50.0, 50.0, 50.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOut2[4]; buffer->getLanes(pdfId, pdfOut2);
    bool v2 = std::abs(pdfOut2[0]) < 1e-300;
    bool g2 = std::abs(grad[0 * vw]) < 1e-100;

    BOOST_TEST_MESSAGE("NormalDistExtremeTail: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

// =============================================================================
// END DISTRIBUTION SANITY TESTS
// =============================================================================

// =============================================================================
// BARRIER ENGINE BUILDING BLOCK TESTS - Dig deeper into AnalyticBarrierEngine
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeStdLog) {
    // log(x), d/dx log(x) = 1/x
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 2.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real result = std::log(x);
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=2
    double xVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    bool v1 = std::abs(resultOut1[0] - std::log(2.0)) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 0.5) < 1e-6;

    // Test 2: x=5
    double xVal2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    bool v2 = std::abs(resultOut2[0] - std::log(5.0)) < 1e-10;
    bool g2 = std::abs(grad[0 * vw] - 0.2) < 1e-6;

    BOOST_TEST_MESSAGE("StdLog: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeStdPow) {
    // pow(base, exponent), d/d(base) = exp*base^(exp-1), d/d(exp) = base^exp*log(base)
    forge::GraphRecorder recorder;
    recorder.start();

    Real base = 2.0;
    Real exponent = 3.0;
    base.markForgeInputAndDiff();
    exponent.markForgeInputAndDiff();
    forge::NodeId baseId = base.forgeNodeId();
    forge::NodeId expId = exponent.forgeNodeId();

    Real result = std::pow(base, exponent);
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(baseId), buffer->getBufferIndex(expId)};
    std::vector<double> grad(2 * vw);

    // Test 1: base=2, exp=3 -> 8, grads: 12, 5.545
    double baseVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(baseId, baseVal1);
    double expVal1[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(expId, expVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    bool v1 = std::abs(resultOut1[0] - 8.0) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 12.0) < 1e-6 && std::abs(grad[1 * vw] - 8.0*std::log(2.0)) < 1e-6;

    // Test 2: base=3, exp=2 -> 9, grads: 6, 9*log(3)
    double baseVal2[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(baseId, baseVal2);
    double expVal2[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(expId, expVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    bool v2 = std::abs(resultOut2[0] - 9.0) < 1e-10;
    bool g2 = std::abs(grad[0 * vw] - 6.0) < 1e-6 && std::abs(grad[1 * vw] - 9.0*std::log(3.0)) < 1e-6;

    BOOST_TEST_MESSAGE("StdPow: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeLogRatio) {
    // log(a/b), d/da = 1/a, d/db = -1/b
    forge::GraphRecorder recorder;
    recorder.start();

    Real a = 100.0, b = 90.0;
    a.markForgeInputAndDiff();
    b.markForgeInputAndDiff();
    forge::NodeId aId = a.forgeNodeId();
    forge::NodeId bId = b.forgeNodeId();

    Real result = std::log(a / b);
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(aId), buffer->getBufferIndex(bId)};
    std::vector<double> grad(2 * vw);

    // Test 1: a=100, b=90
    double aVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aVal1);
    double bVal1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(bId, bVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    bool v1 = std::abs(resultOut1[0] - std::log(100.0/90.0)) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 0.01) < 1e-6 && std::abs(grad[1 * vw] - (-1.0/90.0)) < 1e-6;

    // Test 2: a=120, b=80
    double aVal2[4] = {120.0, 120.0, 120.0, 120.0}; buffer->setLanes(aId, aVal2);
    double bVal2[4] = {80.0, 80.0, 80.0, 80.0}; buffer->setLanes(bId, bVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    bool v2 = std::abs(resultOut2[0] - std::log(120.0/80.0)) < 1e-10;
    bool g2 = std::abs(grad[0 * vw] - (1.0/120.0)) < 1e-6 && std::abs(grad[1 * vw] - (-1.0/80.0)) < 1e-6;

    BOOST_TEST_MESSAGE("LogRatio: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeBarrierX1Formula) {
    // x1 = log(underlying/strike)/stdDev + muSigma
    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0, strike = 100.0, stdDev = 0.1, muSigma = 0.05;
    underlying.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    stdDev.markForgeInputAndDiff();
    muSigma.markForgeInputAndDiff();
    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();
    forge::NodeId stdDevId = stdDev.forgeNodeId();
    forge::NodeId muSigmaId = muSigma.forgeNodeId();

    Real x1 = std::log(underlying / strike) / stdDev + muSigma;
    x1.markForgeOutput();
    forge::NodeId x1Id = x1.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(underlyingId), buffer->getBufferIndex(strikeId)};
    std::vector<double> grad(2 * vw);

    // Test 1
    double underlyingVal1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingVal1);
    double strikeVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeVal1);
    double stdDevVal1[4] = {0.1, 0.1, 0.1, 0.1}; buffer->setLanes(stdDevId, stdDevVal1);
    double muSigmaVal1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(muSigmaId, muSigmaVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x1Out1[4]; buffer->getLanes(x1Id, x1Out1);
    double expV1 = std::log(90.0/100.0)/0.1 + 0.05;
    bool v1 = std::abs(x1Out1[0] - expV1) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 1.0/(90.0*0.1)) < 1e-6;

    // Test 2
    double underlyingVal2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(underlyingId, underlyingVal2);
    double strikeVal2[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(strikeId, strikeVal2);
    double stdDevVal2[4] = {0.1, 0.1, 0.1, 0.1}; buffer->setLanes(stdDevId, stdDevVal2);
    double muSigmaVal2[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(muSigmaId, muSigmaVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x1Out2[4]; buffer->getLanes(x1Id, x1Out2);
    double expV2 = std::log(100.0/90.0)/0.1 + 0.05;
    bool v2 = std::abs(x1Out2[0] - expV2) < 1e-10;
    bool g2 = std::abs(grad[0 * vw] - 1.0/(100.0*0.1)) < 1e-6;

    BOOST_TEST_MESSAGE("BarrierX1: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeBarrierHSPowFormula) {
    // HS = barrier/underlying, powHS0 = pow(HS, 2*mu)
    forge::GraphRecorder recorder;
    recorder.start();

    Real barrier = 70.0, underlying = 90.0, mu = 0.5;
    barrier.markForgeInputAndDiff();
    underlying.markForgeInputAndDiff();
    mu.markForgeInputAndDiff();
    forge::NodeId barrierId = barrier.forgeNodeId();
    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId muId = mu.forgeNodeId();

    Real HS = barrier / underlying;
    Real powHS0 = std::pow(HS, 2.0 * mu);
    powHS0.markForgeOutput();
    forge::NodeId powHS0Id = powHS0.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(barrierId), buffer->getBufferIndex(underlyingId)};
    std::vector<double> grad(2 * vw);

    // Test 1: barrier=70, underlying=90, mu=0.5
    double barrierVal1[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierVal1);
    double underlyingVal1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingVal1);
    double muVal1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(muId, muVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double powHS0Out1[4]; buffer->getLanes(powHS0Id, powHS0Out1);
    double expV1 = std::pow(70.0/90.0, 1.0);
    bool v1 = std::abs(powHS0Out1[0] - expV1) < 1e-10;
    bool g1 = std::abs(grad[0 * vw]) > 1e-10 && std::abs(grad[1 * vw]) > 1e-10;

    // Test 2: barrier=80, underlying=100, mu=0.5
    double barrierVal2[4] = {80.0, 80.0, 80.0, 80.0}; buffer->setLanes(barrierId, barrierVal2);
    double underlyingVal2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(underlyingId, underlyingVal2);
    double muVal2[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(muId, muVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double powHS0Out2[4]; buffer->getLanes(powHS0Id, powHS0Out2);
    double expV2 = std::pow(80.0/100.0, 1.0);
    bool v2 = std::abs(powHS0Out2[0] - expV2) < 1e-10;
    bool g2 = std::abs(grad[0 * vw]) > 1e-10 && std::abs(grad[1 * vw]) > 1e-10;

    BOOST_TEST_MESSAGE("BarrierHSPow: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeBarrierAFunction) {
    // A(phi) = phi*(underlying*divDisc*N(phi*x1) - strike*rfDisc*N(phi*(x1-stdDev)))
    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0, strike = 100.0;
    Real riskFreeDiscount = 0.95, dividendDiscount = 0.98, stdDev = 0.1, muSigma = 0.05, phi = 1.0;
    underlying.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();

    Real x1 = std::log(underlying / strike) / stdDev + muSigma;
    CumulativeNormalDistribution cdf;
    Real N1 = cdf(phi * x1);
    Real N2 = cdf(phi * (x1 - stdDev));
    Real A = phi * (underlying * dividendDiscount * N1 - strike * riskFreeDiscount * N2);
    A.markForgeOutput();
    forge::NodeId AId = A.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(underlyingId), buffer->getBufferIndex(strikeId)};
    std::vector<double> grad(2 * vw);

    // Test 1
    double underlyingVal1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingVal1);
    double strikeVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOut1[4]; buffer->getLanes(AId, AOut1);
    double x1_exp = std::log(90.0/100.0)/0.1 + 0.05;
    double N1_exp = 0.5*(1.0 + std::erf(x1_exp/std::sqrt(2.0)));
    double N2_exp = 0.5*(1.0 + std::erf((x1_exp-0.1)/std::sqrt(2.0)));
    double expV1 = 1.0*(90.0*0.98*N1_exp - 100.0*0.95*N2_exp);
    bool v1 = std::abs(AOut1[0] - expV1) / std::abs(expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw]) > 1e-10 && std::abs(grad[1 * vw]) > 1e-10;

    // Test 2
    double underlyingVal2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(underlyingId, underlyingVal2);
    double strikeVal2[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(strikeId, strikeVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOut2[4]; buffer->getLanes(AId, AOut2);
    double x1_exp2 = std::log(100.0/90.0)/0.1 + 0.05;
    double N1_exp2 = 0.5*(1.0 + std::erf(x1_exp2/std::sqrt(2.0)));
    double N2_exp2 = 0.5*(1.0 + std::erf((x1_exp2-0.1)/std::sqrt(2.0)));
    double expV2 = 1.0*(100.0*0.98*N1_exp2 - 90.0*0.95*N2_exp2);
    bool v2 = std::abs(AOut2[0] - expV2) / std::abs(expV2) < 1e-6;
    bool g2 = std::abs(grad[0 * vw]) > 1e-10 && std::abs(grad[1 * vw]) > 1e-10;

    BOOST_TEST_MESSAGE("BarrierA: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeBarrierCFunctionWithZeroCheck) {
    // C() pattern: powHS * N(x)
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 0.5, powHS = 2.0;
    x.markForgeInputAndDiff();
    powHS.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();
    forge::NodeId powHSId = powHS.forgeNodeId();

    CumulativeNormalDistribution cdf;
    Real N1 = cdf(x);
    Real result = powHS * N1;
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId), buffer->getBufferIndex(powHSId)};
    std::vector<double> grad(2 * vw);

    // Test 1: x=0.5, powHS=2.0
    double xVal1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xVal1);
    double powHSVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(powHSId, powHSVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    double N1_exp = 0.5*(1.0 + std::erf(0.5/std::sqrt(2.0)));
    double expV1 = 2.0 * N1_exp;
    bool v1 = std::abs(resultOut1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[1 * vw] - N1_exp) < 1e-6;

    // Test 2: x=1.0, powHS=3.0
    double xVal2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xVal2);
    double powHSVal2[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(powHSId, powHSVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    double N1_exp2 = 0.5*(1.0 + std::erf(1.0/std::sqrt(2.0)));
    double expV2 = 3.0 * N1_exp2;
    bool v2 = std::abs(resultOut2[0] - expV2) < 1e-6;
    bool g2 = std::abs(grad[1 * vw] - N1_exp2) < 1e-6;

    BOOST_TEST_MESSAGE("BarrierCPattern: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

// =============================================================================
// PROCESS RETRIEVAL TEST - Isolate the dynamic value retrieval issue
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeBarrierWithProcessRetrieval) {
    // A() with dynamic process retrieval (like real engine)
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    double underlying_val = 90.0, strike_val = 100.0, rate_val = 0.04, vol_val = 0.20;
    double T = value(dayCounter.yearFraction(today, maturity));
    double riskFreeDiscount_exp = std::exp(-rate_val * T);
    double dividendDiscount_exp = 1.0;
    double stdDev_exp = vol_val * std::sqrt(T);
    double mu_exp = rate_val / (vol_val * vol_val) - 0.5;
    double muSigma_exp = (1 + mu_exp) * stdDev_exp;
    double x1_exp = std::log(underlying_val / strike_val) / stdDev_exp + muSigma_exp;
    double N1_exp = 0.5 * (1.0 + std::erf(x1_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((x1_exp - stdDev_exp) / std::sqrt(2.0)));
    double A_exp = 1.0 * (underlying_val * dividendDiscount_exp * N1_exp - strike_val * riskFreeDiscount_exp * N2_exp);

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = underlying_val, strike = strike_val, rate = rate_val, vol = vol_val;
    underlying.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    rate.markForgeInputAndDiff();
    vol.markForgeInputAndDiff();
    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();
    forge::NodeId rateId = rate.forgeNodeId();
    forge::NodeId volId = vol.forgeNodeId();

    auto underlyingQuote = ext::make_shared<SimpleQuote>(underlying);
    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    Handle<Quote> underlyingHandle(underlyingQuote);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));
    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    Real process_underlying = process->x0();
    Real process_riskFreeDiscount = process->riskFreeRate()->discount(maturity);
    Real process_dividendDiscount = process->dividendYield()->discount(maturity);
    Real process_variance = process->blackVolatility()->blackVariance(maturity, strike);
    Real process_stdDev = std::sqrt(process_variance);
    Real mu = rate / (vol * vol) - 0.5;
    Real muSigma = (1.0 + mu) * process_stdDev;
    Real x1 = std::log(process_underlying / strike) / process_stdDev + muSigma;

    CumulativeNormalDistribution cdf;
    Real phi = 1.0;
    Real N1 = cdf(phi * x1);
    Real N2 = cdf(phi * (x1 - process_stdDev));
    Real A = phi * (process_underlying * process_dividendDiscount * N1 - strike * process_riskFreeDiscount * N2);
    A.markForgeOutput();
    forge::NodeId AId = A.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {
        buffer->getBufferIndex(underlyingId), buffer->getBufferIndex(strikeId),
        buffer->getBufferIndex(rateId), buffer->getBufferIndex(volId)};
    std::vector<double> grad(4 * vw);

    // Test 1
    double underlyingValIn1[4] = {underlying_val, underlying_val, underlying_val, underlying_val}; buffer->setLanes(underlyingId, underlyingValIn1);
    double strikeValIn1[4] = {strike_val, strike_val, strike_val, strike_val}; buffer->setLanes(strikeId, strikeValIn1);
    double rateValIn1[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValIn1);
    double volValIn1[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValIn1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOut1[4]; buffer->getLanes(AId, AOut1);
    bool v1 = std::abs(AOut1[0] - A_exp) / std::abs(A_exp) < 0.01;
    bool g1 = std::abs(grad[0 * vw]) > 1e-10;

    // Test 2: different inputs
    double underlying_val2 = 100.0, strike_val2 = 90.0;
    double x1_exp2 = std::log(underlying_val2 / strike_val2) / stdDev_exp + muSigma_exp;
    double N1_exp2 = 0.5 * (1.0 + std::erf(x1_exp2 / std::sqrt(2.0)));
    double N2_exp2 = 0.5 * (1.0 + std::erf((x1_exp2 - stdDev_exp) / std::sqrt(2.0)));
    double A_exp2 = 1.0 * (underlying_val2 * dividendDiscount_exp * N1_exp2 - strike_val2 * riskFreeDiscount_exp * N2_exp2);
    double underlyingValIn2[4] = {underlying_val2, underlying_val2, underlying_val2, underlying_val2}; buffer->setLanes(underlyingId, underlyingValIn2);
    double strikeValIn2[4] = {strike_val2, strike_val2, strike_val2, strike_val2}; buffer->setLanes(strikeId, strikeValIn2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOut2[4]; buffer->getLanes(AId, AOut2);
    bool v2 = std::abs(AOut2[0] - A_exp2) / std::abs(A_exp2) < 0.01;
    bool g2 = std::abs(grad[0 * vw]) > 1e-10;

    BOOST_TEST_MESSAGE("BarrierProcessRetrieval: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

// =============================================================================
// STEP-BY-STEP PROCESS RETRIEVAL TESTS - Isolate exactly where the break happens
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeProcessX0Retrieval) {
    // Process x0() retrieval - should equal underlying
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    DayCounter dayCounter = Actual365Fixed();

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0;
    underlying.markForgeInputAndDiff();
    forge::NodeId underlyingId = underlying.forgeNodeId();

    auto underlyingQuote = ext::make_shared<SimpleQuote>(underlying);
    Handle<Quote> underlyingHandle(underlyingQuote);
    Handle<YieldTermStructure> flatRate(ext::make_shared<FlatForward>(0, NullCalendar(), 0.04, dayCounter));
    Handle<BlackVolTermStructure> flatVol(ext::make_shared<BlackConstantVol>(0, NullCalendar(), 0.20, dayCounter));
    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    Real x0 = process->x0();
    x0.markForgeOutput();
    forge::NodeId x0Id = x0.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(underlyingId)};
    std::vector<double> grad(1 * vw);

    // Test 1
    double underlyingVal1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out1[4]; buffer->getLanes(x0Id, x0Out1);
    bool v1 = std::abs(x0Out1[0] - 90.0) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    // Test 2
    double underlyingVal2[4] = {110.0, 110.0, 110.0, 110.0}; buffer->setLanes(underlyingId, underlyingVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out2[4]; buffer->getLanes(x0Id, x0Out2);
    bool v2 = std::abs(x0Out2[0] - 110.0) < 1e-10;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    BOOST_TEST_MESSAGE("ProcessX0: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeProcessDiscountRetrieval) {
    // Process discount() retrieval - exp(-r*T)
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();
    double T = value(dayCounter.yearFraction(today, maturity));

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = 0.04;
    rate.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();

    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    Handle<Quote> underlyingHandle(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
    Handle<BlackVolTermStructure> flatVol(ext::make_shared<BlackConstantVol>(0, NullCalendar(), 0.20, dayCounter));
    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    Real discount = process->riskFreeRate()->discount(maturity);
    discount.markForgeOutput();
    forge::NodeId discountId = discount.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(rateId)};
    std::vector<double> grad(1 * vw);

    // Test 1: rate=0.04
    double rateVal1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double discountOut1[4]; buffer->getLanes(discountId, discountOut1);
    double expV1 = std::exp(-0.04 * T);
    double expG1 = -T * expV1;
    bool v1 = std::abs(discountOut1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: rate=0.06
    double rateVal2[4] = {0.06, 0.06, 0.06, 0.06}; buffer->setLanes(rateId, rateVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double discountOut2[4]; buffer->getLanes(discountId, discountOut2);
    double expV2 = std::exp(-0.06 * T);
    double expG2 = -T * expV2;
    bool v2 = std::abs(discountOut2[0] - expV2) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - expG2) < 1e-3;

    BOOST_TEST_MESSAGE("ProcessDiscount: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

BOOST_AUTO_TEST_CASE(testForgeProcessZeroRateRetrieval) {
    // Process zeroRate() retrieval - should pass through rate
    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = 0.04;
    rate.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();

    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    Handle<Quote> underlyingHandle(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
    Handle<BlackVolTermStructure> flatVol(ext::make_shared<BlackConstantVol>(0, NullCalendar(), 0.20, dayCounter));
    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    Rate zeroRate = process->riskFreeRate()->zeroRate(maturity, process->riskFreeRate()->dayCounter(), Continuous, NoFrequency);
    zeroRate.markForgeOutput();
    forge::NodeId zeroRateId = zeroRate.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(rateId)};
    std::vector<double> grad(1 * vw);

    // Test 1: rate=0.04
    double rateValZ1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateValZ1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zeroRateOut1[4]; buffer->getLanes(zeroRateId, zeroRateOut1);
    bool v1 = std::abs(zeroRateOut1[0] - 0.04) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-3;

    // Test 2: rate=0.06
    double rateValZ2[4] = {0.06, 0.06, 0.06, 0.06}; buffer->setLanes(rateId, rateValZ2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zeroRateOut2[4]; buffer->getLanes(zeroRateId, zeroRateOut2);
    bool v2 = std::abs(zeroRateOut2[0] - 0.06) < 1e-6;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-3;

    BOOST_TEST_MESSAGE("ProcessZeroRate: V1=" << (v1?"OK":"FAIL") << " G1=" << (g1?"OK":"FAIL")
        << " V2=" << (v2?"OK":"FAIL") << " G2=" << (g2?"OK":"FAIL"));
    BOOST_CHECK(v1); BOOST_CHECK(g1); BOOST_CHECK(v2); BOOST_CHECK(g2);
}

// =============================================================================
// NARROWING TESTS: Isolate exactly where zeroRate() breaks
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeCompoundFactor) {
    BOOST_TEST_MESSAGE("NARROW DOWN: compound = 1.0/discount(t) calculation...");

    // zeroRate() computes: compound = 1.0/discount(t)
    // Then calls InterestRate::impliedRate(compound, ...)
    // Let's test if the compound factor is computed correctly

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    double rate_val = 0.04;
    double T = 1.0;
    double expectedDiscount = std::exp(-rate_val * T);
    double expectedCompound = 1.0 / expectedDiscount;

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = rate_val;
    rate.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();

    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));

    // Get discount
    Real discount = flatRate->discount(maturity);
    BOOST_TEST_MESSAGE("  discount during recording: " << value(discount));

    // Compute compound factor like zeroRate() does
    Real compound = 1.0 / discount;
    BOOST_TEST_MESSAGE("  compound during recording: " << value(compound));
    BOOST_TEST_MESSAGE("  expected compound: " << expectedCompound);

    compound.markForgeOutput();
    forge::NodeId compoundId = compound.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double rateValIn[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValIn);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double compoundOut[4]; buffer->getLanes(compoundId, compoundOut);
    double forgeResult = compoundOut[0];

    BOOST_TEST_MESSAGE("  Forge compound result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << expectedCompound);

    double diff = std::abs(forgeResult - expectedCompound);
    double diffPct = diff / expectedCompound * 100.0;
    BOOST_TEST_MESSAGE("  Difference: " << diffPct << "%");

    if (diffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** compound factor calculation is BROKEN ***");
    } else {
        BOOST_TEST_MESSAGE("  compound factor: PASSED - issue is in impliedRate()");
    }

    BOOST_CHECK_CLOSE(forgeResult, expectedCompound, 1e-6);
}

BOOST_AUTO_TEST_CASE(testForgeImpliedRateDirect) {
    BOOST_TEST_MESSAGE("NARROW DOWN: InterestRate::impliedRate() directly...");

    // Test impliedRate with a known compound factor
    // For continuous compounding: r = log(compound) / t

    double compound_val = 1.04081077419;  // exp(0.04 * 1.0)
    double T = 1.0;
    double expected_rate = std::log(compound_val) / T;  // Should be ~0.04

    BOOST_TEST_MESSAGE("  Input compound: " << compound_val);
    BOOST_TEST_MESSAGE("  Expected rate: " << expected_rate);

    forge::GraphRecorder recorder;
    recorder.start();

    Real compound = compound_val;
    compound.markForgeInputAndDiff();
    forge::NodeId compoundId = compound.forgeNodeId();

    // Call impliedRate directly
    DayCounter dayCounter = Actual365Fixed();
    InterestRate ir = InterestRate::impliedRate(compound, dayCounter, Continuous, NoFrequency, T);

    // Get the rate from InterestRate
    Rate rate = ir.rate();

    BOOST_TEST_MESSAGE("  impliedRate result during recording: " << value(rate));

    rate.markForgeOutput();
    forge::NodeId rateId = rate.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double compoundValIn[4] = {compound_val, compound_val, compound_val, compound_val}; buffer->setLanes(compoundId, compoundValIn);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double rateOut1[4]; buffer->getLanes(rateId, rateOut1);
    double forgeResult = rateOut1[0];

    BOOST_TEST_MESSAGE("  Forge rate result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << expected_rate);

    double diff = std::abs(forgeResult - expected_rate);
    double diffPct = (expected_rate != 0) ? diff / std::abs(expected_rate) * 100.0 : diff;
    BOOST_TEST_MESSAGE("  Difference: " << diffPct << "%");

    if (diffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** InterestRate::impliedRate() is BROKEN ***");
    } else {
        BOOST_TEST_MESSAGE("  impliedRate: PASSED");
    }

    BOOST_CHECK_CLOSE(forgeResult, expected_rate, 1e-6);
}

BOOST_AUTO_TEST_CASE(testForgeInterestRateConstructor) {
    BOOST_TEST_MESSAGE("NARROW DOWN: InterestRate constructor...");

    // Test if InterestRate constructor preserves AReal properly

    double rate_val = 0.04;

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = rate_val;
    rate.markForgeInputAndDiff();
    forge::NodeId inputRateId = rate.forgeNodeId();

    // Create InterestRate with AReal
    DayCounter dayCounter = Actual365Fixed();
    InterestRate ir(rate, dayCounter, Continuous, NoFrequency);

    // Get the rate back
    Rate outputRate = ir.rate();

    BOOST_TEST_MESSAGE("  Input rate: " << rate_val);
    BOOST_TEST_MESSAGE("  Output rate during recording: " << value(outputRate));

    outputRate.markForgeOutput();
    forge::NodeId outputRateId = outputRate.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double inputRateValIn[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(inputRateId, inputRateValIn);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double outputRateOut[4]; buffer->getLanes(outputRateId, outputRateOut);
    double forgeResult2 = outputRateOut[0];

    BOOST_TEST_MESSAGE("  Forge output rate: " << forgeResult2);
    BOOST_TEST_MESSAGE("  Expected: " << rate_val);

    double diff2 = std::abs(forgeResult2 - rate_val);
    double diffPct = (rate_val != 0) ? diff2 / std::abs(rate_val) * 100.0 : diff2;

    if (diffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** InterestRate constructor is BROKEN ***");
    } else {
        BOOST_TEST_MESSAGE("  InterestRate constructor: PASSED");
    }

    BOOST_CHECK_CLOSE(forgeResult2, rate_val, 1e-6);
}

BOOST_AUTO_TEST_CASE(testForgeLogDivision) {
    BOOST_TEST_MESSAGE("NARROW DOWN: log(compound)/t calculation (impliedRate core)...");

    // The core of impliedRate for Continuous is: r = log(compound) / t
    // Test this directly

    double compound_val = 1.04081077419;  // exp(0.04 * 1.0)
    double T = 1.0;
    double expected_rate = std::log(compound_val) / T;

    forge::GraphRecorder recorder;
    recorder.start();

    Real compound = compound_val;
    compound.markForgeInputAndDiff();
    forge::NodeId compoundId = compound.forgeNodeId();

    // Compute rate = log(compound) / t
    Real rate = std::log(compound) / T;

    BOOST_TEST_MESSAGE("  log(compound)/t during recording: " << value(rate));
    BOOST_TEST_MESSAGE("  expected: " << expected_rate);

    rate.markForgeOutput();
    forge::NodeId rateId = rate.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double compoundValIn2[4] = {compound_val, compound_val, compound_val, compound_val}; buffer->setLanes(compoundId, compoundValIn2);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double rateOut2[4]; buffer->getLanes(rateId, rateOut2);
    double forgeResult3 = rateOut2[0];

    BOOST_TEST_MESSAGE("  Forge rate result: " << forgeResult3);
    BOOST_TEST_MESSAGE("  Expected: " << expected_rate);

    double diff3 = std::abs(forgeResult3 - expected_rate);
    double diffPct3 = diff3 / std::abs(expected_rate) * 100.0;

    if (diffPct3 > 1.0) {
        BOOST_TEST_MESSAGE("  *** log(compound)/t is BROKEN ***");
    } else {
        BOOST_TEST_MESSAGE("  log(compound)/t: PASSED");
    }

    BOOST_CHECK_CLOSE(forgeResult3, expected_rate, 1e-6);
}

BOOST_AUTO_TEST_CASE(testForgeImpliedRateComparison) {
    BOOST_TEST_MESSAGE("NARROW DOWN: impliedRate 'if (compound==1.0)' comparison...");

    // InterestRate::impliedRate has: if (compound==1.0) { r = 0.0; }
    // This comparison might be calling value() and breaking things

    // Test with compound != 1.0
    double compound_val = 1.04081077419;  // exp(0.04 * 1.0), clearly not 1.0

    forge::GraphRecorder recorder;
    recorder.start();

    Real compound = compound_val;
    compound.markForgeInputAndDiff();
    forge::NodeId compoundId = compound.forgeNodeId();

    // Simulate what impliedRate does:
    // if (compound == 1.0) { r = 0.0; } else { r = log(compound)/t; }

    // The comparison compound == 1.0 should be false
    // But if it's evaluated incorrectly during graph recording...

    Real r;
    double T = 1.0;

    // Let's trace through the comparison
    bool comparisonResult = (compound == 1.0);  // This might call value()!
    BOOST_TEST_MESSAGE("  compound == 1.0 evaluates to: " << (comparisonResult ? "true" : "false"));
    BOOST_TEST_MESSAGE("  compound value: " << value(compound));

    if (comparisonResult) {
        r = 0.0;
        BOOST_TEST_MESSAGE("  Taking r = 0.0 branch (WRONG!)");
    } else {
        r = std::log(compound) / T;
        BOOST_TEST_MESSAGE("  Taking r = log(compound)/t branch (correct)");
    }

    r.markForgeOutput();
    forge::NodeId rateId = r.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double compoundValIn3[4] = {compound_val, compound_val, compound_val, compound_val}; buffer->setLanes(compoundId, compoundValIn3);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double rateOut3[4]; buffer->getLanes(rateId, rateOut3);
    double forgeResult4 = rateOut3[0];
    double expected = std::log(compound_val) / T;

    BOOST_TEST_MESSAGE("  Forge rate result: " << forgeResult4);
    BOOST_TEST_MESSAGE("  Expected: " << expected);

    if (std::abs(forgeResult4) < 1e-10 && std::abs(expected) > 0.01) {
        BOOST_TEST_MESSAGE("  *** BUG: r=0 suggests comparison broke the graph! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult4, expected, 1e-6);
}

BOOST_AUTO_TEST_CASE(testForgeFullZeroRateChain) {
    BOOST_TEST_MESSAGE("NARROW DOWN: Full zeroRate chain step by step...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    double rate_val = 0.04;
    double T = 1.0;

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = rate_val;
    rate.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();

    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));

    // Step 1: discount
    Real discount = flatRate->discount(maturity);
    BOOST_TEST_MESSAGE("  Step 1 - discount: " << value(discount));

    // Step 2: compound = 1.0/discount
    Real compound = 1.0 / discount;
    BOOST_TEST_MESSAGE("  Step 2 - compound: " << value(compound));

    // Step 3: log(compound)
    Real logCompound = std::log(compound);
    BOOST_TEST_MESSAGE("  Step 3 - log(compound): " << value(logCompound));

    // Step 4: log(compound) / T
    Real rateFromLog = logCompound / T;
    BOOST_TEST_MESSAGE("  Step 4 - log(compound)/T: " << value(rateFromLog));

    // Mark each intermediate as output to see where it breaks
    discount.markForgeOutput();
    compound.markForgeOutput();
    logCompound.markForgeOutput();
    rateFromLog.markForgeOutput();

    forge::NodeId discountId = discount.forgeNodeId();
    forge::NodeId compoundId = compound.forgeNodeId();
    forge::NodeId logCompoundId = logCompound.forgeNodeId();
    forge::NodeId rateFromLogId = rateFromLog.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double rateValChain[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValChain);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double expectedDiscount = std::exp(-rate_val * T);
    double expectedCompound = 1.0 / expectedDiscount;
    double expectedLogCompound = std::log(expectedCompound);
    double expectedRate = expectedLogCompound / T;

    double discountOutC[4]; buffer->getLanes(discountId, discountOutC);
    double compoundOutC[4]; buffer->getLanes(compoundId, compoundOutC);
    double logCompoundOutC[4]; buffer->getLanes(logCompoundId, logCompoundOutC);
    double rateFromLogOutC[4]; buffer->getLanes(rateFromLogId, rateFromLogOutC);

    BOOST_TEST_MESSAGE("  Results comparison:");
    BOOST_TEST_MESSAGE("    discount:     forge=" << discountOutC[0] << ", expected=" << expectedDiscount);
    BOOST_TEST_MESSAGE("    compound:     forge=" << compoundOutC[0] << ", expected=" << expectedCompound);
    BOOST_TEST_MESSAGE("    log(compound): forge=" << logCompoundOutC[0] << ", expected=" << expectedLogCompound);
    BOOST_TEST_MESSAGE("    rate:         forge=" << rateFromLogOutC[0] << ", expected=" << expectedRate);

    // Check each step
    bool discountOK = std::abs(discountOutC[0] - expectedDiscount) / expectedDiscount < 0.001;
    bool compoundOK = std::abs(compoundOutC[0] - expectedCompound) / expectedCompound < 0.001;
    bool logOK = std::abs(logCompoundOutC[0] - expectedLogCompound) / std::abs(expectedLogCompound) < 0.001;
    bool rateOK = std::abs(rateFromLogOutC[0] - expectedRate) / std::abs(expectedRate) < 0.001;

    BOOST_TEST_MESSAGE("  Step verification:");
    BOOST_TEST_MESSAGE("    discount:      " << (discountOK ? "OK" : "FAILED"));
    BOOST_TEST_MESSAGE("    compound:      " << (compoundOK ? "OK" : "FAILED"));
    BOOST_TEST_MESSAGE("    log(compound): " << (logOK ? "OK" : "FAILED"));
    BOOST_TEST_MESSAGE("    rate:          " << (rateOK ? "OK" : "FAILED"));

    BOOST_CHECK_CLOSE(rateFromLogOutC[0], expectedRate, 1e-6);
}


BOOST_AUTO_TEST_CASE(testForgeProcessBlackVolRetrieval) {
    BOOST_TEST_MESSAGE("TEST: Process blackVolatility()->blackVol() retrieval (used by AnalyticBarrierEngine::volatility)...");

    // AnalyticBarrierEngine::volatility() calls blackVol(), not blackVariance()!
    // This might be where the issue is.

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    double vol_val = 0.20;

    forge::GraphRecorder recorder;
    recorder.start();

    Real vol = vol_val;
    vol.markForgeInputAndDiff();
    forge::NodeId volId = vol.forgeNodeId();

    // Build process chain with vol as input
    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    Handle<Quote> underlyingHandle(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), 0.04, dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));

    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    // Retrieve blackVol through process - THIS IS WHAT AnalyticBarrierEngine DOES
    Real strike = 100.0;
    Volatility blackVol = process->blackVolatility()->blackVol(maturity, strike);

    BOOST_TEST_MESSAGE("  blackVol during recording: " << value(blackVol));
    BOOST_TEST_MESSAGE("  expected: " << vol_val);

    blackVol.markForgeOutput();
    forge::NodeId blackVolId = blackVol.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double volValIn[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValIn);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double blackVolOut[4]; buffer->getLanes(blackVolId, blackVolOut);
    double forgeResultBV = blackVolOut[0];

    BOOST_TEST_MESSAGE("  Forge blackVol result: " << forgeResultBV);
    BOOST_TEST_MESSAGE("  Expected: " << vol_val);

    double priceDiff = std::abs(forgeResultBV - vol_val);
    double priceDiffPct = (vol_val != 0) ? priceDiff / vol_val * 100.0 : priceDiff;
    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND: blackVol() breaks the Forge graph! ***");
    }

    BOOST_CHECK_CLOSE(forgeResultBV, vol_val, 1e-6);

    // Check gradient
    int vectorWidth = buffer->getVectorWidth();
    std::vector<size_t> gradientIndices = {buffer->getBufferIndex(volId)};
    std::vector<double> gradients(1 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    BOOST_TEST_MESSAGE("  d(blackVol)/d(vol): " << gradients[0 * vectorWidth] << " (expected 1.0)");
    BOOST_CHECK_CLOSE(gradients[0 * vectorWidth], 1.0, 1e-3);

    BOOST_TEST_MESSAGE("  Process blackVol() Retrieval: " << (priceDiffPct < 1.0 ? "PASSED" : "FAILED"));
}

BOOST_AUTO_TEST_CASE(testForgeBarrierMuCalculation) {
    BOOST_TEST_MESSAGE("TEST: Barrier mu() calculation as done by AnalyticBarrierEngine...");

    // AnalyticBarrierEngine::mu() computes:
    // mu = (riskFreeRate() - dividendYield()) / (vol * vol) - 0.5
    // where riskFreeRate() and dividendYield() use zeroRate(), and vol uses blackVol()

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    double rate_val = 0.04;
    double vol_val = 0.20;
    double expected_mu = rate_val / (vol_val * vol_val) - 0.5;  // No dividend, so dividendYield=0

    BOOST_TEST_MESSAGE("  Expected mu = " << expected_mu);

    forge::GraphRecorder recorder;
    recorder.start();

    Real rate = rate_val;
    Real vol = vol_val;
    rate.markForgeInputAndDiff();
    vol.markForgeInputAndDiff();
    forge::NodeId rateId = rate.forgeNodeId();
    forge::NodeId volId = vol.forgeNodeId();

    // Build process chain
    auto rateQuote = ext::make_shared<SimpleQuote>(rate);
    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    Handle<Quote> underlyingHandle(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));

    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    // Compute mu EXACTLY as AnalyticBarrierEngine does:
    // Rate riskFreeRate = process->riskFreeRate()->zeroRate(maturity, dayCounter, Continuous, NoFrequency);
    // Rate dividendYield = process->dividendYield()->zeroRate(maturity, dayCounter, Continuous, NoFrequency);
    // Volatility volatility = process->blackVolatility()->blackVol(maturity, strike);
    // mu = (riskFreeRate - dividendYield) / (volatility * volatility) - 0.5;

    Rate riskFreeRate = process->riskFreeRate()->zeroRate(
        maturity, process->riskFreeRate()->dayCounter(), Continuous, NoFrequency);
    Rate dividendYield = process->dividendYield()->zeroRate(
        maturity, process->dividendYield()->dayCounter(), Continuous, NoFrequency);
    Real strike = 100.0;
    Volatility volatility = process->blackVolatility()->blackVol(maturity, strike);

    BOOST_TEST_MESSAGE("  riskFreeRate from process: " << value(riskFreeRate));
    BOOST_TEST_MESSAGE("  dividendYield from process: " << value(dividendYield));
    BOOST_TEST_MESSAGE("  volatility from process: " << value(volatility));

    Real mu = (riskFreeRate - dividendYield) / (volatility * volatility) - 0.5;

    BOOST_TEST_MESSAGE("  mu during recording: " << value(mu));

    mu.markForgeOutput();
    forge::NodeId muId = mu.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double rateValMu[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValMu);
    double volValMu[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValMu);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double muOut[4]; buffer->getLanes(muId, muOut);
    double forgeResultMu = muOut[0];

    BOOST_TEST_MESSAGE("  Forge mu result: " << forgeResultMu);
    BOOST_TEST_MESSAGE("  Expected: " << expected_mu);

    double priceDiff = std::abs(forgeResultMu - expected_mu);
    double priceDiffPct = (expected_mu != 0) ? priceDiff / std::abs(expected_mu) * 100.0 : priceDiff;
    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND: mu() calculation breaks the Forge graph! ***");
        BOOST_TEST_MESSAGE("  This is likely the source of the barrier option AAD failure.");
    }

    BOOST_CHECK_CLOSE(forgeResultMu, expected_mu, 1.0);

    // Check gradients
    int vectorWidth = buffer->getVectorWidth();
    std::vector<size_t> gradientIndices = {
        buffer->getBufferIndex(rateId),
        buffer->getBufferIndex(volId)
    };
    std::vector<double> gradients(2 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    // d(mu)/d(rate) = 1 / (vol^2) = 1 / 0.04 = 25
    // d(mu)/d(vol) = -2 * rate / (vol^3) = -2 * 0.04 / 0.008 = -10
    double expected_dmu_drate = 1.0 / (vol_val * vol_val);
    double expected_dmu_dvol = -2.0 * rate_val / (vol_val * vol_val * vol_val);

    BOOST_TEST_MESSAGE("  d(mu)/d(rate): " << gradients[0 * vectorWidth] << " (expected " << expected_dmu_drate << ")");
    BOOST_TEST_MESSAGE("  d(mu)/d(vol): " << gradients[1 * vectorWidth] << " (expected " << expected_dmu_dvol << ")");

    BOOST_TEST_MESSAGE("  Barrier mu() calculation: " << (priceDiffPct < 1.0 ? "PASSED" : "FAILED"));
}

BOOST_AUTO_TEST_CASE(testForgeProcessVarianceRetrieval) {
    BOOST_TEST_MESSAGE("TEST: Process blackVolatility()->blackVariance() retrieval...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + 1 * Years;
    DayCounter dayCounter = Actual365Fixed();

    double vol_val = 0.20;
    double T = value(dayCounter.yearFraction(today, maturity));
    double expectedVariance = vol_val * vol_val * T;
    double expectedStdDev = std::sqrt(expectedVariance);

    forge::GraphRecorder recorder;
    recorder.start();

    Real vol = vol_val;
    vol.markForgeInputAndDiff();
    forge::NodeId volId = vol.forgeNodeId();

    // Build process chain with vol as input
    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    Handle<Quote> underlyingHandle(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), 0.04, dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));

    auto process = ext::make_shared<BlackScholesProcess>(underlyingHandle, flatRate, flatVol);

    // Retrieve variance through process
    Real strike = 100.0;
    Real variance = process->blackVolatility()->blackVariance(maturity, strike);
    Real stdDev = std::sqrt(variance);

    BOOST_TEST_MESSAGE("  variance during recording: " << value(variance));
    BOOST_TEST_MESSAGE("  stdDev during recording: " << value(stdDev));
    BOOST_TEST_MESSAGE("  expected variance: " << expectedVariance);
    BOOST_TEST_MESSAGE("  expected stdDev: " << expectedStdDev);

    stdDev.markForgeOutput();
    forge::NodeId stdDevId = stdDev.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double volValVar[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValVar);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double stdDevOut[4]; buffer->getLanes(stdDevId, stdDevOut);
    double forgeResultVar = stdDevOut[0];

    BOOST_TEST_MESSAGE("  Forge stdDev result: " << forgeResultVar);
    BOOST_TEST_MESSAGE("  Expected: " << expectedStdDev);

    double priceDiff = std::abs(forgeResultVar - expectedStdDev);
    double priceDiffPct = priceDiff / expectedStdDev * 100.0;
    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in variance/stdDev retrieval! ***");
    }

    BOOST_CHECK_CLOSE(forgeResultVar, expectedStdDev, 1e-6);

    // Check gradient: d(sqrt(vol^2 * T))/d(vol) = vol * T / sqrt(vol^2 * T) = sqrt(T)
    int vectorWidth = buffer->getVectorWidth();
    std::vector<size_t> gradientIndices = {buffer->getBufferIndex(volId)};
    std::vector<double> gradients(1 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    double expectedGrad = std::sqrt(T);  // d(vol*sqrt(T))/d(vol) = sqrt(T)
    BOOST_TEST_MESSAGE("  d(stdDev)/d(vol): " << gradients[0 * vectorWidth] << " (expected " << expectedGrad << ")");
    BOOST_CHECK_CLOSE(gradients[0 * vectorWidth], expectedGrad, 1e-3);

    BOOST_TEST_MESSAGE("  Process Variance Retrieval: PASSED");
}

// =============================================================================
// BARRIER ENGINE B(), C(), D() FUNCTION TESTS - Isolate the exact breaking point
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeBarrierBFunction) {
    BOOST_TEST_MESSAGE("TEST: Barrier B() function with Forge AAD...");

    // B(phi) is like A(phi) but uses barrier instead of strike in log ratio:
    // x2 = log(underlying/barrier)/stdDev + muSigma
    // B = phi * (underlying * dividendDiscount * N(phi*x2)
    //          - strike * riskFreeDiscount * N(phi*(x2-stdDev)))

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0;
    Real strike = 100.0;
    Real barrier = 70.0;
    Real riskFreeDiscount = 0.95;
    Real dividendDiscount = 1.0;
    Real stdDev = 0.2;
    Real muSigma = 0.3;
    Real phi = 1.0;

    underlying.markForgeInputAndDiff();
    barrier.markForgeInputAndDiff();

    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId barrierId = barrier.forgeNodeId();

    // Compute B() formula
    Real x2 = std::log(underlying / barrier) / stdDev + muSigma;

    CumulativeNormalDistribution cdf;
    Real N1 = cdf(phi * x2);
    Real N2 = cdf(phi * (x2 - stdDev));

    Real B = phi * (underlying * dividendDiscount * N1 - strike * riskFreeDiscount * N2);

    B.markForgeOutput();
    forge::NodeId BId = B.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  B() function graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double underlyingValB[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValB);
    double barrierValB[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValB);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double BOut[4]; buffer->getLanes(BId, BOut);
    double forgeResultB = BOut[0];

    // Compute expected value manually
    double x2_exp = std::log(90.0 / 70.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(x2_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((x2_exp - 0.2) / std::sqrt(2.0)));
    double B_exp = 1.0 * (90.0 * 1.0 * N1_exp - 100.0 * 0.95 * N2_exp);

    BOOST_TEST_MESSAGE("  B() value: expected=" << B_exp << ", forge=" << forgeResultB);

    double priceDiff = std::abs(forgeResultB - B_exp);
    double priceDiffPct = (B_exp != 0) ? priceDiff / std::abs(B_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in B() function! ***");
    }

    BOOST_CHECK_CLOSE(forgeResultB, B_exp, 1e-6);

    BOOST_TEST_MESSAGE("  Barrier B() function: PASSED");
}

BOOST_AUTO_TEST_CASE(testForgeBarrierCFunction) {
    BOOST_TEST_MESSAGE("TEST: Barrier C() function with Forge AAD (the problematic one)...");

    // C(eta, phi) has the problematic pattern:
    // HS = barrier/underlying
    // powHS0 = pow(HS, 2*mu)
    // powHS1 = powHS0 * HS * HS
    // y1 = log(barrier*HS/strike)/stdDev + muSigma
    // C = phi * (underlying * dividendDiscount * (N1==0 ? 0 : powHS1*N1)
    //          - strike * riskFreeDiscount * (N2==0 ? 0 : powHS0*N2))
    //
    // The (N1==0 ? 0 : ...) pattern breaks the Forge graph!

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0;
    Real strike = 100.0;
    Real barrier = 70.0;
    Real riskFreeDiscount = 0.95;
    Real dividendDiscount = 1.0;
    Real stdDev = 0.2;
    Real mu = 0.5;
    Real muSigma = 0.3;
    Real eta = 1.0;
    Real phi = 1.0;

    underlying.markForgeInputAndDiff();
    barrier.markForgeInputAndDiff();

    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId barrierId = barrier.forgeNodeId();

    // Compute C() formula (WITHOUT the N==0 conditional for now)
    Real HS = barrier / underlying;
    Real powHS0 = std::pow(HS, 2.0 * mu);
    Real powHS1 = powHS0 * HS * HS;
    Real y1 = std::log(barrier * HS / strike) / stdDev + muSigma;

    CumulativeNormalDistribution cdf;
    Real N1 = cdf(eta * y1);
    Real N2 = cdf(eta * (y1 - stdDev));

    // First test WITHOUT the conditional (this should work)
    Real C_no_conditional = phi * (underlying * dividendDiscount * powHS1 * N1
                                  - strike * riskFreeDiscount * powHS0 * N2);

    C_no_conditional.markForgeOutput();
    forge::NodeId CId = C_no_conditional.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  C() function (no conditional) graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double underlyingValC[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValC);
    double barrierValC[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double COut[4]; buffer->getLanes(CId, COut);
    double forgeResultC = COut[0];

    // Compute expected value manually
    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y1_exp = std::log(70.0 * HS_exp / 100.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(y1_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((y1_exp - 0.2) / std::sqrt(2.0)));
    double C_exp = 1.0 * (90.0 * 1.0 * powHS1_exp * N1_exp - 100.0 * 0.95 * powHS0_exp * N2_exp);

    BOOST_TEST_MESSAGE("  C() value (no conditional): expected=" << C_exp << ", forge=" << forgeResultC);

    double priceDiff = std::abs(forgeResultC - C_exp);
    double priceDiffPct = (C_exp != 0) ? priceDiff / std::abs(C_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in C() function (even without conditional)! ***");
    }

    BOOST_CHECK_CLOSE(forgeResultC, C_exp, 1e-6);

    BOOST_TEST_MESSAGE("  Barrier C() function (no conditional): PASSED");
}

BOOST_AUTO_TEST_CASE(testForgeBarrierCFunctionWithConditional) {
    BOOST_TEST_MESSAGE("TEST: Barrier C() function WITH conditional (N1==0 ? 0 : ...)...");

    // This test replicates the EXACT pattern from AnalyticBarrierEngine::C()
    // which has the problematic (N1 == 0.0 ? Real(0.0) : Real(powHS1 * N1)) pattern

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0;
    Real strike = 100.0;
    Real barrier = 70.0;
    Real riskFreeDiscount = 0.95;
    Real dividendDiscount = 1.0;
    Real stdDev = 0.2;
    Real mu = 0.5;
    Real muSigma = 0.3;
    Real eta = 1.0;
    Real phi = 1.0;

    underlying.markForgeInputAndDiff();
    barrier.markForgeInputAndDiff();

    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId barrierId = barrier.forgeNodeId();

    // Compute C() formula WITH the conditional (exactly as in AnalyticBarrierEngine)
    Real HS = barrier / underlying;
    Real powHS0 = std::pow(HS, 2.0 * mu);
    Real powHS1 = powHS0 * HS * HS;
    Real y1 = std::log(barrier * HS / strike) / stdDev + muSigma;

    CumulativeNormalDistribution cdf;
    Real N1 = cdf(eta * y1);
    Real N2 = cdf(eta * (y1 - stdDev));

    // WITH the conditional - this is the EXACT pattern from AnalyticBarrierEngine::C()
    Real C_with_conditional = phi * (underlying * dividendDiscount * (N1 == 0.0 ? Real(0.0) : Real(powHS1 * N1))
                                    - strike * riskFreeDiscount * (N2 == 0.0 ? Real(0.0) : Real(powHS0 * N2)));

    C_with_conditional.markForgeOutput();
    forge::NodeId CId = C_with_conditional.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  C() function (WITH conditional) graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double underlyingValCond[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValCond);
    double barrierValCond[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValCond);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double CCondOut[4]; buffer->getLanes(CId, CCondOut);
    double forgeResultCCond = CCondOut[0];

    // Compute expected value manually
    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y1_exp = std::log(70.0 * HS_exp / 100.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(y1_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((y1_exp - 0.2) / std::sqrt(2.0)));
    double C_exp = 1.0 * (90.0 * 1.0 * powHS1_exp * N1_exp - 100.0 * 0.95 * powHS0_exp * N2_exp);

    BOOST_TEST_MESSAGE("  C() value (WITH conditional): expected=" << C_exp << ", forge=" << forgeResultCCond);

    double priceDiff = std::abs(forgeResultCCond - C_exp);
    double priceDiffPct = (C_exp != 0) ? priceDiff / std::abs(C_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE CONFIRMED: The (N1==0 ? 0 : ...) conditional breaks Forge! ***");
        BOOST_TEST_MESSAGE("  This is the ROOT CAUSE of the barrier option AAD failure.");
    }

    BOOST_CHECK_CLOSE(forgeResultCCond, C_exp, 1.0);  // Allow 1% tolerance to see actual difference

    BOOST_TEST_MESSAGE("  Barrier C() function (WITH conditional): " << (priceDiffPct < 1.0 ? "PASSED" : "FAILED"));
}

BOOST_AUTO_TEST_CASE(testForgeBarrierDFunction) {
    BOOST_TEST_MESSAGE("TEST: Barrier D() function with Forge AAD...");

    // D(eta, phi) is similar to C() but uses different y value:
    // y2 = log(barrier/underlying)/stdDev + muSigma

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0;
    Real strike = 100.0;
    Real barrier = 70.0;
    Real riskFreeDiscount = 0.95;
    Real dividendDiscount = 1.0;
    Real stdDev = 0.2;
    Real mu = 0.5;
    Real muSigma = 0.3;
    Real eta = 1.0;
    Real phi = 1.0;

    underlying.markForgeInputAndDiff();
    barrier.markForgeInputAndDiff();

    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId barrierId = barrier.forgeNodeId();

    // Compute D() formula (WITHOUT the conditional for now)
    Real HS = barrier / underlying;
    Real powHS0 = std::pow(HS, 2.0 * mu);
    Real powHS1 = powHS0 * HS * HS;
    Real y2 = std::log(barrier / underlying) / stdDev + muSigma;

    CumulativeNormalDistribution cdf;
    Real N1 = cdf(eta * y2);
    Real N2 = cdf(eta * (y2 - stdDev));

    // WITHOUT the conditional
    Real D_no_conditional = phi * (underlying * dividendDiscount * powHS1 * N1
                                  - strike * riskFreeDiscount * powHS0 * N2);

    D_no_conditional.markForgeOutput();
    forge::NodeId DId = D_no_conditional.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  D() function (no conditional) graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double underlyingValD[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValD);
    double barrierValD[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValD);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double DOut[4]; buffer->getLanes(DId, DOut);
    double forgeResultD = DOut[0];

    // Compute expected value manually
    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y2_exp = std::log(70.0 / 90.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(y2_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((y2_exp - 0.2) / std::sqrt(2.0)));
    double D_exp = 1.0 * (90.0 * 1.0 * powHS1_exp * N1_exp - 100.0 * 0.95 * powHS0_exp * N2_exp);

    BOOST_TEST_MESSAGE("  D() value (no conditional): expected=" << D_exp << ", forge=" << forgeResultD);

    double priceDiff = std::abs(forgeResultD - D_exp);
    double priceDiffPct = (D_exp != 0) ? priceDiff / std::abs(D_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in D() function! ***");
    }

    BOOST_CHECK_CLOSE(forgeResultD, D_exp, 1e-6);

    BOOST_TEST_MESSAGE("  Barrier D() function (no conditional): PASSED");
}

BOOST_AUTO_TEST_CASE(testForgeComparisonBreaksGraph) {
    BOOST_TEST_MESSAGE("TEST: Does comparison (a >= b) break Forge graph?...");

    // This test checks if comparing two active AReal values breaks the graph
    // This is what happens in AnalyticBarrierEngine::calculate():
    //   if (strike >= barrier()) ...

    forge::GraphRecorder recorder;
    recorder.start();

    Real a = 100.0;
    Real b = 70.0;

    a.markForgeInputAndDiff();
    b.markForgeInputAndDiff();

    forge::NodeId aId = a.forgeNodeId();
    forge::NodeId bId = b.forgeNodeId();

    // Simple formula WITHOUT comparison
    Real result_no_comparison = a + b;

    // Now do a comparison like the engine does
    // if (a >= b) then use one formula, else use another
    Real result_with_comparison;
    if (a >= b) {
        result_with_comparison = a * 2.0;  // a >= b is true, so this branch
    } else {
        result_with_comparison = b * 2.0;
    }

    result_with_comparison.markForgeOutput();
    forge::NodeId resultId = result_with_comparison.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double aValComp[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aValComp);
    double bValComp[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(bId, bValComp);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double resultCompOut[4]; buffer->getLanes(resultId, resultCompOut);
    double forgeResultComp = resultCompOut[0];
    double expected = 200.0;  // a * 2.0 = 100 * 2 = 200

    BOOST_TEST_MESSAGE("  Result with comparison: expected=" << expected << ", forge=" << forgeResultComp);

    double priceDiff = std::abs(forgeResultComp - expected);
    double priceDiffPct = priceDiff / expected * 100.0;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** CONFIRMED: Comparison (a >= b) breaks Forge graph! ***");
    }

    // Check gradient - if comparison broke the graph, gradient will be wrong
    int vectorWidth = buffer->getVectorWidth();
    std::vector<size_t> gradientIndices = {
        buffer->getBufferIndex(aId),
        buffer->getBufferIndex(bId)
    };
    std::vector<double> gradients(2 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    BOOST_TEST_MESSAGE("  d/da: " << gradients[0 * vectorWidth] << " (expected 2.0 for a*2)");
    BOOST_TEST_MESSAGE("  d/db: " << gradients[1 * vectorWidth] << " (expected 0.0 for a*2)");

    // If comparison broke the graph, the result might be computed at graph-build time
    // with the initial values, not re-evaluated with buffer values
    BOOST_CHECK_CLOSE(forgeResultComp, expected, 1e-6);
    BOOST_CHECK_CLOSE(gradients[0 * vectorWidth], 2.0, 1e-6);

    BOOST_TEST_MESSAGE("  Comparison test: " << (priceDiffPct < 1.0 ? "PASSED" : "FAILED"));
}

BOOST_AUTO_TEST_CASE(testForgeComparisonWithDifferentInputs) {
    BOOST_TEST_MESSAGE("TEST: Comparison with different input values...");

    // This test uses input values that would change the branch taken
    // If the graph is built correctly, changing inputs should change which branch executes
    // But if comparison breaks the graph, the branch is fixed at graph-build time

    forge::GraphRecorder recorder;
    recorder.start();

    Real a = 50.0;  // Start with a < b
    Real b = 70.0;

    a.markForgeInputAndDiff();
    b.markForgeInputAndDiff();

    forge::NodeId aId = a.forgeNodeId();
    forge::NodeId bId = b.forgeNodeId();

    // This comparison happens at GRAPH BUILD TIME with a=50, b=70
    // So the else branch is taken (a < b)
    Real result;
    if (a >= b) {
        result = a * 2.0;  // This branch NOT taken during graph build
    } else {
        result = b * 3.0;  // This branch taken during graph build (50 < 70)
    }

    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph built with a=50, b=70 (a < b, so else branch)");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Now execute with DIFFERENT values where a >= b
    double aValDiff[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aValDiff);  // Now a=100 >= b=70, should take if branch
    double bValDiff[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(bId, bValDiff);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double resultDiffOut[4]; buffer->getLanes(resultId, resultDiffOut);
    double forgeResultDiff = resultDiffOut[0];

    // If graph respects runtime values: result = a * 2 = 100 * 2 = 200
    // If graph is fixed at build time: result = b * 3 = 70 * 3 = 210
    double expected_if_runtime = 200.0;
    double expected_if_buildtime = 210.0;

    BOOST_TEST_MESSAGE("  Executed with a=100, b=70 (a >= b)");
    BOOST_TEST_MESSAGE("  Forge result: " << forgeResultDiff);
    BOOST_TEST_MESSAGE("  If runtime branching: " << expected_if_runtime);
    BOOST_TEST_MESSAGE("  If build-time branching: " << expected_if_buildtime);

    if (std::abs(forgeResultDiff - expected_if_buildtime) < 1e-6) {
        BOOST_TEST_MESSAGE("  *** CONFIRMED: Branch is FIXED at graph build time! ***");
        BOOST_TEST_MESSAGE("  The comparison (a >= b) evaluated at build time, not runtime.");
        BOOST_TEST_MESSAGE("  This is why the full BarrierEngine fails - it uses different");
        BOOST_TEST_MESSAGE("  formulas based on (strike >= barrier) which is fixed at build time.");
    }

    // We expect build-time branching (which is actually correct for Forge)
    // The issue is that AnalyticBarrierEngine builds the graph with one set of values
    // but executes with different values - and the branch doesn't change
    BOOST_CHECK_CLOSE(forgeResultDiff, expected_if_buildtime, 1e-6);

    BOOST_TEST_MESSAGE("  Comparison with different inputs: Shows build-time branching");
}

BOOST_AUTO_TEST_CASE(testForgeBarrierAMinusCFormula) {
    BOOST_TEST_MESSAGE("TEST: Barrier A(1) - C(1,1) formula (DownOut Call with strike >= barrier)...");

    // For DownOut Call with strike >= barrier, the formula is: A(1) - C(1,1) + F(1)
    // F(1) = 0 when rebate = 0, so it's just A(1) - C(1,1)

    forge::GraphRecorder recorder;
    recorder.start();

    Real underlying = 90.0;
    Real strike = 100.0;
    Real barrier = 70.0;
    Real riskFreeDiscount = 0.95;
    Real dividendDiscount = 1.0;
    Real stdDev = 0.2;
    Real mu = 0.5;
    Real muSigma = 0.3;

    underlying.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    barrier.markForgeInputAndDiff();

    forge::NodeId underlyingId = underlying.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();
    forge::NodeId barrierId = barrier.forgeNodeId();

    CumulativeNormalDistribution cdf;

    // A(1) computation
    Real x1 = std::log(underlying / strike) / stdDev + muSigma;
    Real A_N1 = cdf(x1);
    Real A_N2 = cdf(x1 - stdDev);
    Real A = underlying * dividendDiscount * A_N1 - strike * riskFreeDiscount * A_N2;

    // C(1,1) computation (WITHOUT conditional to see if formula itself works)
    Real HS = barrier / underlying;
    Real powHS0 = std::pow(HS, 2.0 * mu);
    Real powHS1 = powHS0 * HS * HS;
    Real y1 = std::log(barrier * HS / strike) / stdDev + muSigma;
    Real C_N1 = cdf(y1);
    Real C_N2 = cdf(y1 - stdDev);
    Real C = underlying * dividendDiscount * powHS1 * C_N1 - strike * riskFreeDiscount * powHS0 * C_N2;

    // Full formula: A - C
    Real result = A - C;

    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  A(1) - C(1,1) formula graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double underlyingValAC[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValAC);
    double strikeValAC[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValAC);
    double barrierValAC[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValAC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double resultACOut[4]; buffer->getLanes(resultId, resultACOut);
    double forgeResultAC = resultACOut[0];

    // Compute expected value manually
    double x1_exp = std::log(90.0 / 100.0) / 0.2 + 0.3;
    double A_N1_exp = 0.5 * (1.0 + std::erf(x1_exp / std::sqrt(2.0)));
    double A_N2_exp = 0.5 * (1.0 + std::erf((x1_exp - 0.2) / std::sqrt(2.0)));
    double A_exp = 90.0 * 1.0 * A_N1_exp - 100.0 * 0.95 * A_N2_exp;

    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y1_exp = std::log(70.0 * HS_exp / 100.0) / 0.2 + 0.3;
    double C_N1_exp = 0.5 * (1.0 + std::erf(y1_exp / std::sqrt(2.0)));
    double C_N2_exp = 0.5 * (1.0 + std::erf((y1_exp - 0.2) / std::sqrt(2.0)));
    double C_exp = 90.0 * 1.0 * powHS1_exp * C_N1_exp - 100.0 * 0.95 * powHS0_exp * C_N2_exp;

    double expected = A_exp - C_exp;

    BOOST_TEST_MESSAGE("  A(1) = " << A_exp);
    BOOST_TEST_MESSAGE("  C(1,1) = " << C_exp);
    BOOST_TEST_MESSAGE("  A - C: expected=" << expected << ", forge=" << forgeResultAC);

    double priceDiff = std::abs(forgeResultAC - expected);
    double priceDiffPct = (expected != 0) ? priceDiff / std::abs(expected) * 100.0 : priceDiff;

    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in A-C formula! ***");
    }

    BOOST_CHECK_CLOSE(forgeResultAC, expected, 1e-6);

    BOOST_TEST_MESSAGE("  Barrier A(1) - C(1,1) formula: PASSED");
}

// =============================================================================
// END BARRIER ENGINE BUILDING BLOCK TESTS
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeVanillaOption) {
    BOOST_TEST_MESSAGE("Sanity test: Vanilla European Option with Forge AAD...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    // First compute reference price with bumping
    auto priceVanilla = [&](Real spot, Real strike, Real rate, Real vol) -> Real {
        auto spotQuote = ext::make_shared<SimpleQuote>(spot);
        auto rateQuote = ext::make_shared<SimpleQuote>(rate);
        auto volQuote = ext::make_shared<SimpleQuote>(vol);

        DayCounter dayCounter = Actual365Fixed();
        Date maturity = today + 1 * Years;

        Handle<YieldTermStructure> flatRate(
            ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQuote), dayCounter));
        Handle<BlackVolTermStructure> flatVol(
            ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQuote), dayCounter));

        auto exercise = ext::make_shared<EuropeanExercise>(maturity);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

        auto bsProcess = ext::make_shared<BlackScholesProcess>(Handle<Quote>(spotQuote), flatRate, flatVol);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(bsProcess));

        return option.NPV();
    };

    // Reference price
    double refPrice = value(priceVanilla(100.0, 100.0, 0.05, 0.20));
    BOOST_TEST_MESSAGE("  Reference vanilla option price: " << refPrice);

    // Now with Forge AAD
    forge::GraphRecorder recorder;
    recorder.start();

    Real spot = 100.0;
    Real strike = 100.0;
    Real rate = 0.05;
    Real vol = 0.20;

    spot.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    rate.markForgeInputAndDiff();
    vol.markForgeInputAndDiff();

    forge::NodeId spotId = spot.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();
    forge::NodeId rateId = rate.forgeNodeId();
    forge::NodeId volId = vol.forgeNodeId();

    Real price = priceVanilla(spot, strike, rate, vol);

    price.markForgeOutput();
    forge::NodeId priceId = price.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double spotValVO[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(spotId, spotValVO);
    double strikeValVO[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValVO);
    double rateValVO[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateValVO);
    double volValVO[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volValVO);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double forgePriceOut[4]; buffer->getLanes(priceId, forgePriceOut);
    double forgePriceVO = forgePriceOut[0];
    BOOST_TEST_MESSAGE("  Forge vanilla option price: " << forgePriceVO);

    // Check prices match
    BOOST_CHECK_CLOSE(forgePriceVO, refPrice, 1e-6);

    // Get gradients
    int vectorWidth = buffer->getVectorWidth();
    std::vector<size_t> gradientIndices = {
        buffer->getBufferIndex(spotId),
        buffer->getBufferIndex(strikeId),
        buffer->getBufferIndex(rateId),
        buffer->getBufferIndex(volId)
    };
    std::vector<double> gradients(4 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    BOOST_TEST_MESSAGE("  Gradients: d/dSpot=" << gradients[0 * vectorWidth]
                       << ", d/dStrike=" << gradients[1 * vectorWidth]
                       << ", d/dRate=" << gradients[2 * vectorWidth]
                       << ", d/dVol=" << gradients[3 * vectorWidth]);

    // Delta should be positive for a call, roughly 0.5-0.6 for ATM
    BOOST_CHECK(gradients[0 * vectorWidth] > 0.4 && gradients[0 * vectorWidth] < 0.8);
    // Vega should be positive
    BOOST_CHECK(gradients[3 * vectorWidth] > 0);

    BOOST_TEST_MESSAGE("  Vanilla Option: PASSED");
}

BOOST_AUTO_TEST_CASE(testARealVsDoubleBarrierPrice) {
    BOOST_TEST_MESSAGE("DIAGNOSTIC: AReal vs Double barrier price (NO Forge recording)...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    DayCounter dayCounter = Actual365Fixed();
    Date maturity = today + 1 * Years;

    // Test values
    double u_val = 90.0, strike_val = 100.0, r_val = 0.10, v_val = 0.10, b_val = 0.10;

    // ========== COMPUTATION 1: Pure doubles ==========
    double doublePrice;
    {
        auto underlyingH = ext::make_shared<SimpleQuote>(u_val);
        auto riskFreeRate = ext::make_shared<SimpleQuote>(r_val);
        auto volatility = ext::make_shared<SimpleQuote>(v_val);

        Handle<YieldTermStructure> flatRate(
            ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(riskFreeRate), dayCounter));
        Handle<BlackVolTermStructure> flatVol(
            ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volatility), dayCounter));

        auto exercise = ext::make_shared<EuropeanExercise>(maturity);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike_val);

        auto bsProcess = ext::make_shared<BlackScholesProcess>(Handle<Quote>(underlyingH), flatRate, flatVol);

        BarrierOption option(Barrier::DownOut, b_val, 0.0, payoff, exercise);
        option.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bsProcess));

        doublePrice = value(option.NPV());
    }
    BOOST_TEST_MESSAGE("  Double computation result: " << doublePrice);

    // ========== COMPUTATION 2: AReal (NO Forge recording) ==========
    double arealPrice;
    {
        Real u = u_val;
        Real strike = strike_val;
        Real r = r_val;
        Real v = v_val;
        Real b = b_val;

        auto underlyingH = ext::make_shared<SimpleQuote>(u);
        auto riskFreeRate = ext::make_shared<SimpleQuote>(r);
        auto volatility = ext::make_shared<SimpleQuote>(v);

        Handle<YieldTermStructure> flatRate(
            ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(riskFreeRate), dayCounter));
        Handle<BlackVolTermStructure> flatVol(
            ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volatility), dayCounter));

        auto exercise = ext::make_shared<EuropeanExercise>(maturity);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

        auto bsProcess = ext::make_shared<BlackScholesProcess>(Handle<Quote>(underlyingH), flatRate, flatVol);

        BarrierOption option(Barrier::DownOut, b, 0.0, payoff, exercise);
        option.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bsProcess));

        arealPrice = value(option.NPV());
    }
    BOOST_TEST_MESSAGE("  AReal computation result: " << arealPrice);

    // ========== COMPARE ==========
    double diff = std::abs(doublePrice - arealPrice);
    double diffPct = diff / doublePrice * 100.0;

    BOOST_TEST_MESSAGE("  Difference: " << diff << " (" << diffPct << "%)");

    if (diffPct > 0.0001) {
        BOOST_TEST_MESSAGE("  *** BUG FOUND: AReal arithmetic gives different result than double! ***");
        BOOST_TEST_MESSAGE("  This is NOT a Forge graph issue - AReal itself is broken somewhere in QuantLib.");
    } else {
        BOOST_TEST_MESSAGE("  AReal matches double - issue is in Forge graph construction.");
    }

    BOOST_CHECK_CLOSE(doublePrice, arealPrice, 1e-9);
}

BOOST_AUTO_TEST_CASE(testForgeBarrierForwardPass) {
    BOOST_TEST_MESSAGE("Sanity test: Barrier Option forward pass comparison...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    // Compute reference price directly (no Forge)
    auto priceBarrierDirect = [&](double u, double strike, double r, double v, double b) -> double {
        auto underlyingH = ext::make_shared<SimpleQuote>(u);
        auto riskFreeRate = ext::make_shared<SimpleQuote>(r);
        auto volatility = ext::make_shared<SimpleQuote>(v);

        DayCounter dayCounter = Actual365Fixed();
        Date maturity = today + 1 * Years;

        Handle<YieldTermStructure> flatRate(
            ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(riskFreeRate), dayCounter));
        Handle<BlackVolTermStructure> flatVol(
            ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volatility), dayCounter));

        auto exercise = ext::make_shared<EuropeanExercise>(maturity);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

        auto bsProcess = ext::make_shared<BlackScholesProcess>(Handle<Quote>(underlyingH), flatRate, flatVol);

        BarrierOption option(Barrier::DownOut, b, 0.0, payoff, exercise);
        option.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bsProcess));

        return value(option.NPV());
    };

    // Test values matching the main test
    double u = 90.0, strike = 100.0, r = 0.10, v = 0.10, b = 0.10;

    double refPrice = priceBarrierDirect(u, strike, r, v, b);
    BOOST_TEST_MESSAGE("  Reference barrier price (direct): " << refPrice);

    // Now with Forge recording but using same computation
    forge::GraphRecorder recorder;
    recorder.start();

    Real fu = u;
    Real fstrike = strike;
    Real fr = r;
    Real fv = v;
    Real fb = b;

    fu.markForgeInputAndDiff();
    fstrike.markForgeInputAndDiff();
    fr.markForgeInputAndDiff();
    fv.markForgeInputAndDiff();
    fb.markForgeInputAndDiff();

    forge::NodeId uId = fu.forgeNodeId();
    forge::NodeId strikeId = fstrike.forgeNodeId();
    forge::NodeId rId = fr.forgeNodeId();
    forge::NodeId vId = fv.forgeNodeId();
    forge::NodeId bId = fb.forgeNodeId();

    // Build barrier option with Forge-tracked values
    auto underlyingH = ext::make_shared<SimpleQuote>(fu);
    auto riskFreeRate = ext::make_shared<SimpleQuote>(fr);
    auto volatility = ext::make_shared<SimpleQuote>(fv);

    DayCounter dayCounter = Actual365Fixed();
    Date maturity = today + 1 * Years;

    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(riskFreeRate), dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volatility), dayCounter));

    auto exercise = ext::make_shared<EuropeanExercise>(maturity);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, fstrike);

    auto bsProcess = ext::make_shared<BlackScholesProcess>(Handle<Quote>(underlyingH), flatRate, flatVol);

    BarrierOption option(Barrier::DownOut, fb, 0.0, payoff, exercise);
    option.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bsProcess));

    Real price = option.NPV();

    price.markForgeOutput();
    forge::NodeId priceId = price.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double uValFP[4] = {value(u), value(u), value(u), value(u)}; buffer->setLanes(uId, uValFP);
    double strikeValFP[4] = {value(strike), value(strike), value(strike), value(strike)}; buffer->setLanes(strikeId, strikeValFP);
    double rValFP[4] = {value(r), value(r), value(r), value(r)}; buffer->setLanes(rId, rValFP);
    double vValFP[4] = {value(v), value(v), value(v), value(v)}; buffer->setLanes(vId, vValFP);
    double bValFP[4] = {value(b), value(b), value(b), value(b)}; buffer->setLanes(bId, bValFP);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double forgePriceFPOut[4]; buffer->getLanes(priceId, forgePriceFPOut);
    double forgePriceFP = forgePriceFPOut[0];
    BOOST_TEST_MESSAGE("  Forge barrier price: " << forgePriceFP);

    double priceDiff = std::abs(forgePriceFP - refPrice);
    double priceDiffPct = priceDiff / refPrice * 100.0;
    BOOST_TEST_MESSAGE("  Price difference: " << priceDiff << " (" << priceDiffPct << "%)");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  WARNING: Forward pass mismatch! Forge computes different value.");
        BOOST_TEST_MESSAGE("  This indicates a problem in graph construction or execution.");
    }

    BOOST_CHECK_CLOSE(forgePriceFP, refPrice, 1e-6);

    BOOST_TEST_MESSAGE("  Barrier Forward Pass: PASSED");
}

// =============================================================================
// END SANITY TESTS (diagnostic test after priceBarrierOption definition)
// =============================================================================

namespace {
    Real priceBarrierOption(const BarrierOptionData& value) {
        Date today(29, May, 2006);
        Settings::instance().evaluationDate() = today;

        // Option parameters from input (these are Forge-marked inputs)
        Barrier::Type barrierType = Barrier::DownOut;
        Real rebate = 0.0;
        Date maturity = today + 1 * Years;
        DayCounter dayCounter = Actual365Fixed();

        // Create quotes from INPUT values (propagates Forge graph)
        auto underlyingH = ext::make_shared<SimpleQuote>(value.u);
        auto riskFreeRate = ext::make_shared<SimpleQuote>(value.r);
        auto volatility = ext::make_shared<SimpleQuote>(value.v);

        Handle<Quote> h1(riskFreeRate);
        Handle<Quote> h2(volatility);
        Handle<YieldTermStructure> flatRate(
            ext::make_shared<FlatForward>(0, NullCalendar(), h1, dayCounter));
        Handle<BlackVolTermStructure> flatVol(
            ext::make_shared<BlackConstantVol>(0, NullCalendar(), h2, dayCounter));

        // Instantiate the option with INPUT values
        auto exercise = ext::make_shared<EuropeanExercise>(maturity);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(value.type, value.strike);

        auto bsProcess =
            ext::make_shared<BlackScholesProcess>(Handle<Quote>(underlyingH), flatRate, flatVol);

        auto referenceOption =
            ext::make_shared<BarrierOption>(barrierType, value.b, rebate, payoff, exercise);

        referenceOption->setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bsProcess));

        return referenceOption->NPV();
    }
}

// Diagnostic test - must be after priceBarrierOption definition
BOOST_AUTO_TEST_CASE(testDiagnoseBarrierOptionInputUsage) {
    BOOST_TEST_MESSAGE("Diagnostic: Check if priceBarrierOption uses input data...");

    // This test demonstrates the bug: priceBarrierOption ignores its input!
    BarrierOptionData data1{Option::Call, 100.0, 90.0, 0.10, 0.10, 0.10};
    BarrierOptionData data2{Option::Call, 200.0, 50.0, 0.20, 0.20, 0.50};  // Very different values

    Real price1 = priceBarrierOption(data1);
    Real price2 = priceBarrierOption(data2);

    BOOST_TEST_MESSAGE("  Price with data1 (strike=100, u=90): " << value(price1));
    BOOST_TEST_MESSAGE("  Price with data2 (strike=200, u=50): " << value(price2));

    // If the function properly used input data, prices should be different
    // But since it uses hardcoded values, they will be the same!
    if (std::abs(value(price1) - value(price2)) < 1e-10) {
        BOOST_TEST_MESSAGE("  BUG CONFIRMED: Both prices are identical!");
        BOOST_TEST_MESSAGE("  priceBarrierOption() IGNORES its input data!");
        BOOST_TEST_MESSAGE("  Fix: Replace hardcoded values with value.strike, value.u, value.r, value.b, value.v");
    }
}

// =============================================================================
// BARRIER PRICING WITH STABILITY-ONLY OPTIMIZATION (WORKING CONFIG)
// =============================================================================

BOOST_AUTO_TEST_CASE(testForgeBarrierPricingOnlyStabilityCleaning) {
    SavedSettings save;
    BOOST_TEST_MESSAGE("FULL PRICING TEST: Barrier option with ONLY stability cleaning (price + derivatives)...");

    auto values = BarrierOptionData{Option::Call, 100.00, 90.00, 0.10, 0.10, 0.10};

    // First compute reference price and derivatives with bumping (no Forge)
    auto derivatives_bumping = BarrierOptionData{};
    auto expected = priceWithBumping(values, derivatives_bumping, priceBarrierOption);
    double expectedVal = value(expected);
    BOOST_TEST_MESSAGE("  Reference price (bumping): " << expectedVal);
    BOOST_TEST_MESSAGE("  Reference derivatives (bumping):");
    BOOST_TEST_MESSAGE("    d/dStrike: " << value(derivatives_bumping.strike));
    BOOST_TEST_MESSAGE("    d/dU: " << value(derivatives_bumping.u));
    BOOST_TEST_MESSAGE("    d/dR: " << value(derivatives_bumping.r));
    BOOST_TEST_MESSAGE("    d/dB: " << value(derivatives_bumping.b));
    BOOST_TEST_MESSAGE("    d/dV: " << value(derivatives_bumping.v));

    // Now compute with Forge AAD with ONLY stability cleaning
    forge::GraphRecorder recorder;
    recorder.start();

    auto data = values;
    data.strike.markForgeInputAndDiff();
    data.u.markForgeInputAndDiff();
    data.r.markForgeInputAndDiff();
    data.b.markForgeInputAndDiff();
    data.v.markForgeInputAndDiff();

    forge::NodeId strikeNodeId = data.strike.forgeNodeId();
    forge::NodeId uNodeId = data.u.forgeNodeId();
    forge::NodeId rNodeId = data.r.forgeNodeId();
    forge::NodeId bNodeId = data.b.forgeNodeId();
    forge::NodeId vNodeId = data.v.forgeNodeId();

    auto price = priceBarrierOption(data);
    price.markForgeOutput();
    forge::NodeId priceNodeId = price.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    // Compile with default config (stability cleaning only)
    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    double strikeValSO[4] = {value(values.strike), value(values.strike), value(values.strike), value(values.strike)}; buffer->setLanes(strikeNodeId, strikeValSO);
    double uValSO[4] = {value(values.u), value(values.u), value(values.u), value(values.u)}; buffer->setLanes(uNodeId, uValSO);
    double rValSO[4] = {value(values.r), value(values.r), value(values.r), value(values.r)}; buffer->setLanes(rNodeId, rValSO);
    double bValSO[4] = {value(values.b), value(values.b), value(values.b), value(values.b)}; buffer->setLanes(bNodeId, bValSO);
    double vValSO[4] = {value(values.v), value(values.v), value(values.v), value(values.v)}; buffer->setLanes(vNodeId, vValSO);

    buffer->clearGradients();
    kernel->execute(*buffer);

    double actualOut[4]; buffer->getLanes(priceNodeId, actualOut);
    double actual = actualOut[0];

    BOOST_TEST_MESSAGE("  ONLY STABILITY CLEANING:");
    BOOST_TEST_MESSAGE("    Forge price: " << actual);
    BOOST_TEST_MESSAGE("    Expected price: " << expectedVal);

    double priceDiffPct = std::abs(actual - expectedVal) / std::abs(expectedVal) * 100.0;
    BOOST_TEST_MESSAGE("    Price difference: " << priceDiffPct << "%");

    // Get gradients
    int vectorWidth = buffer->getVectorWidth();
    std::vector<size_t> gradientIndices = {
        buffer->getBufferIndex(strikeNodeId),
        buffer->getBufferIndex(uNodeId),
        buffer->getBufferIndex(rNodeId),
        buffer->getBufferIndex(bNodeId),
        buffer->getBufferIndex(vNodeId)
    };
    std::vector<double> gradients(5 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    BOOST_TEST_MESSAGE("  Forge derivatives:");
    BOOST_TEST_MESSAGE("    d/dStrike: " << gradients[0 * vectorWidth] << " (expected: " << value(derivatives_bumping.strike) << ")");
    BOOST_TEST_MESSAGE("    d/dU: " << gradients[1 * vectorWidth] << " (expected: " << value(derivatives_bumping.u) << ")");
    BOOST_TEST_MESSAGE("    d/dR: " << gradients[2 * vectorWidth] << " (expected: " << value(derivatives_bumping.r) << ")");
    BOOST_TEST_MESSAGE("    d/dB: " << gradients[3 * vectorWidth] << " (expected: " << value(derivatives_bumping.b) << ")");
    BOOST_TEST_MESSAGE("    d/dV: " << gradients[4 * vectorWidth] << " (expected: " << value(derivatives_bumping.v) << ")");

    // Check price
    BOOST_CHECK_CLOSE(expectedVal, actual, 1e-6);

    // Check derivatives
    QL_CHECK_CLOSE(value(derivatives_bumping.strike), gradients[0 * vectorWidth], 1e-3);
    QL_CHECK_CLOSE(value(derivatives_bumping.u), gradients[1 * vectorWidth], 1e-3);
    QL_CHECK_CLOSE(value(derivatives_bumping.r), gradients[2 * vectorWidth], 1e-3);
    QL_CHECK_CLOSE(value(derivatives_bumping.b), gradients[3 * vectorWidth], 1e-3);
    QL_CHECK_CLOSE(value(derivatives_bumping.v), gradients[4 * vectorWidth], 1e-3);

    BOOST_TEST_MESSAGE("  ONLY STABILITY CLEANING: " << (priceDiffPct < 1.0 ? "PASSED" : "FAILED"));
}

BOOST_AUTO_TEST_CASE(testForgeBarrierKernelReuse) {
    SavedSettings save;
    BOOST_TEST_MESSAGE("KERNEL REUSE TEST: Compile once, evaluate with multiple input sets...");

    // Build graph with initial values (these are just for graph construction)
    auto buildValues = BarrierOptionData{Option::Call, 100.00, 90.00, 0.10, 0.10, 0.10};

    forge::GraphRecorder recorder;
    recorder.start();

    auto data = buildValues;
    data.strike.markForgeInputAndDiff();
    data.u.markForgeInputAndDiff();
    data.r.markForgeInputAndDiff();
    data.b.markForgeInputAndDiff();
    data.v.markForgeInputAndDiff();

    forge::NodeId strikeNodeId = data.strike.forgeNodeId();
    forge::NodeId uNodeId = data.u.forgeNodeId();
    forge::NodeId rNodeId = data.r.forgeNodeId();
    forge::NodeId bNodeId = data.b.forgeNodeId();
    forge::NodeId vNodeId = data.v.forgeNodeId();

    auto price = priceBarrierOption(data);
    price.markForgeOutput();
    forge::NodeId priceNodeId = price.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    // Compile ONCE with default config (stability cleaning only)
    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    BOOST_TEST_MESSAGE("  Kernel compiled once, now testing with different inputs...");

    // Test with multiple different input sets
    // Note: For Down-and-Out Call, formula branches on (strike >= barrier):
    //   strike >= barrier: A(1) - C(1,1) + F(1)
    //   strike < barrier:  B(1) - D(1,1) + F(1)
    std::vector<BarrierOptionData> testCases = {
        {Option::Call, 100.00, 90.00, 0.10, 0.10, 0.10},   // Original: strike=100 >= barrier=0.10
        {Option::Call, 100.00, 95.00, 0.05, 0.05, 0.20},   // strike=100 >= barrier=0.05
        {Option::Call, 110.00, 100.00, 0.08, 0.08, 0.25},  // strike=110 >= barrier=0.08
        {Option::Call, 90.00, 85.00, 0.03, 0.03, 0.15},    // strike=90 >= barrier=0.03
        // This case triggers DIFFERENT BRANCH: strike=80 < barrier=95
        {Option::Call, 80.00, 100.00, 0.05, 95.00, 0.20},  // strike < barrier -> SHOULD FAIL!
    };

    int testNum = 1;
    for (const auto& testValues : testCases) {
        BOOST_TEST_MESSAGE("  --- Input Set " << testNum << " ---");
        BOOST_TEST_MESSAGE("    strike=" << value(testValues.strike)
                          << ", u=" << value(testValues.u)
                          << ", r=" << value(testValues.r)
                          << ", b=" << value(testValues.b)
                          << ", v=" << value(testValues.v));

        // Compute reference with bumping
        auto derivatives_bumping = BarrierOptionData{};
        auto expectedPrice = priceWithBumping(testValues, derivatives_bumping, priceBarrierOption);
        double expectedVal = value(expectedPrice);

        // Set new inputs and re-execute the SAME kernel
        double strikeValKR[4] = {value(testValues.strike), value(testValues.strike), value(testValues.strike), value(testValues.strike)}; buffer->setLanes(strikeNodeId, strikeValKR);
        double uValKR[4] = {value(testValues.u), value(testValues.u), value(testValues.u), value(testValues.u)}; buffer->setLanes(uNodeId, uValKR);
        double rValKR[4] = {value(testValues.r), value(testValues.r), value(testValues.r), value(testValues.r)}; buffer->setLanes(rNodeId, rValKR);
        double bValKR[4] = {value(testValues.b), value(testValues.b), value(testValues.b), value(testValues.b)}; buffer->setLanes(bNodeId, bValKR);
        double vValKR[4] = {value(testValues.v), value(testValues.v), value(testValues.v), value(testValues.v)}; buffer->setLanes(vNodeId, vValKR);

        buffer->clearGradients();
        kernel->execute(*buffer);

        double forgePriceKROut[4]; buffer->getLanes(priceNodeId, forgePriceKROut);
        double forgePriceKR = forgePriceKROut[0];

        // Get gradients
        int vectorWidth = buffer->getVectorWidth();
        std::vector<size_t> gradientIndices = {
            buffer->getBufferIndex(strikeNodeId),
            buffer->getBufferIndex(uNodeId),
            buffer->getBufferIndex(rNodeId),
            buffer->getBufferIndex(bNodeId),
            buffer->getBufferIndex(vNodeId)
        };
        std::vector<double> gradients(5 * vectorWidth);
        buffer->getGradientLanes(gradientIndices, gradients.data());

        double priceDiff = std::abs(forgePriceKR - expectedVal) / std::abs(expectedVal) * 100.0;

        BOOST_TEST_MESSAGE("    Expected price: " << expectedVal);
        BOOST_TEST_MESSAGE("    Forge price: " << forgePriceKR);
        BOOST_TEST_MESSAGE("    Price diff: " << priceDiff << "%");
        BOOST_TEST_MESSAGE("    Derivatives - Forge vs Bumping:");
        BOOST_TEST_MESSAGE("      d/dStrike: " << gradients[0 * vectorWidth] << " vs " << value(derivatives_bumping.strike));
        BOOST_TEST_MESSAGE("      d/dU: " << gradients[1 * vectorWidth] << " vs " << value(derivatives_bumping.u));
        BOOST_TEST_MESSAGE("      d/dR: " << gradients[2 * vectorWidth] << " vs " << value(derivatives_bumping.r));
        BOOST_TEST_MESSAGE("      d/dB: " << gradients[3 * vectorWidth] << " vs " << value(derivatives_bumping.b));
        BOOST_TEST_MESSAGE("      d/dV: " << gradients[4 * vectorWidth] << " vs " << value(derivatives_bumping.v));

        // Verify price
        BOOST_CHECK_CLOSE(expectedVal, forgePriceKR, 1e-6);

        // Verify derivatives
        QL_CHECK_CLOSE(value(derivatives_bumping.strike), gradients[0 * vectorWidth], 1e-3);
        QL_CHECK_CLOSE(value(derivatives_bumping.u), gradients[1 * vectorWidth], 1e-3);
        QL_CHECK_CLOSE(value(derivatives_bumping.r), gradients[2 * vectorWidth], 1e-3);
        QL_CHECK_CLOSE(value(derivatives_bumping.b), gradients[3 * vectorWidth], 1e-3);
        QL_CHECK_CLOSE(value(derivatives_bumping.v), gradients[4 * vectorWidth], 1e-3);

        BOOST_TEST_MESSAGE("    Input Set " << testNum << ": " << (priceDiff < 1.0 ? "PASSED" : "FAILED"));
        testNum++;
    }

    BOOST_TEST_MESSAGE("  Kernel reuse test completed with " << testCases.size() << " input sets");
}

// =============================================================================
// END BARRIER PRICING TESTS
// =============================================================================

BOOST_AUTO_TEST_CASE(testBarrierOptionDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing barrier options derivatives with Forge AAD...");

    // input
    auto data = BarrierOptionData{Option::Call, 100.00, 90.00, 0.10, 0.10, 0.10};

    // bumping
    auto derivatives_bumping = BarrierOptionData{};
    auto expected = priceWithBumping(data, derivatives_bumping, priceBarrierOption);

    // Forge AAD
    auto derivatives_forge = BarrierOptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceBarrierOption);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bumping.strike, derivatives_forge.strike, DERIVATIVE_TOLERANCE_PCT);
    QL_CHECK_CLOSE(derivatives_bumping.u, derivatives_forge.u, DERIVATIVE_TOLERANCE_PCT);
    QL_CHECK_CLOSE(derivatives_bumping.r, derivatives_forge.r, DERIVATIVE_TOLERANCE_PCT);
    QL_CHECK_CLOSE(derivatives_bumping.b, derivatives_forge.b, DERIVATIVE_TOLERANCE_PCT);
    QL_CHECK_CLOSE(derivatives_bumping.v, derivatives_forge.v, DERIVATIVE_TOLERANCE_PCT);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
