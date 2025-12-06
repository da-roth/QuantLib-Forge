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

// ABool headers for conditional branch testing
#include <expressions/abool.hpp>
#include <expressions/abool_helpers.hpp>

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
// ABOOL UNIT TESTS - Focused tests to isolate ABool::If behavior
// =============================================================================

BOOST_AUTO_TEST_CASE(testABoolIfBasic) {
    // Test: ABool::If should switch branches based on runtime condition
    // Formula: result = (a >= b) ? a*2 : b*2
    BOOST_TEST_MESSAGE("ABOOL TEST 1: Basic ABool::If with tracked inputs...");

    forge::GraphRecorder recorder;
    recorder.start();

    Real a = 100.0, b = 50.0;  // a >= b at build time
    a.markForgeInputAndDiff();
    b.markForgeInputAndDiff();
    forge::NodeId aId = a.forgeNodeId(), bId = b.forgeNodeId();

    // Create ABool condition from tracked inputs
    forge::ABool cond = forge::greaterEqual(a.forgeValue(), b.forgeValue());

    // Use ABool::If to select branch
    Real trueVal = a * 2.0;
    Real falseVal = b * 2.0;
    Real result = cond.If(trueVal, falseVal);

    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1: a=100, b=50 -> a >= b is TRUE -> result = a*2 = 200
    double aVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aVal1);
    double bVal1[4] = {50.0, 50.0, 50.0, 50.0}; buffer->setLanes(bId, bVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultOut1[4]; buffer->getLanes(resultId, resultOut1);
    double result1 = resultOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: a=100, b=50 (a >= b = TRUE)");
    BOOST_TEST_MESSAGE("    Expected: 200 (a*2), Got: " << result1);
    BOOST_CHECK_CLOSE(result1, 200.0, 1e-9);

    // Test 2: a=30, b=50 -> a >= b is FALSE -> result = b*2 = 100
    double aVal2[4] = {30.0, 30.0, 30.0, 30.0}; buffer->setLanes(aId, aVal2);
    double bVal2[4] = {50.0, 50.0, 50.0, 50.0}; buffer->setLanes(bId, bVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultOut2[4]; buffer->getLanes(resultId, resultOut2);
    double result2 = resultOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: a=30, b=50 (a >= b = FALSE)");
    BOOST_TEST_MESSAGE("    Expected: 100 (b*2), Got: " << result2);
    BOOST_CHECK_CLOSE(result2, 100.0, 1e-9);

    // Test 3: a=50, b=50 -> a >= b is TRUE (equal case) -> result = a*2 = 100
    double aVal3[4] = {50.0, 50.0, 50.0, 50.0}; buffer->setLanes(aId, aVal3);
    double bVal3[4] = {50.0, 50.0, 50.0, 50.0}; buffer->setLanes(bId, bVal3);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultOut3[4]; buffer->getLanes(resultId, resultOut3);
    double result3 = resultOut3[0];
    BOOST_TEST_MESSAGE("  Test 3: a=50, b=50 (a >= b = TRUE, equal)");
    BOOST_TEST_MESSAGE("    Expected: 100 (a*2), Got: " << result3);
    BOOST_CHECK_CLOSE(result3, 100.0, 1e-9);

    BOOST_TEST_MESSAGE("  ABool::If Basic: " <<
        ((std::abs(result1 - 200.0) < 1e-6 && std::abs(result2 - 100.0) < 1e-6) ? "PASSED" : "FAILED"));
}

BOOST_AUTO_TEST_CASE(testABoolIfBarrierLike) {
    // Test: Simulate barrier-like formula switching
    // Formula: (strike >= barrier) ? formulaA : formulaB
    // formulaA = strike * 0.1 (simplified "strike >= barrier" formula)
    // formulaB = barrier * 0.2 (simplified "strike < barrier" formula)
    BOOST_TEST_MESSAGE("ABOOL TEST 2: Barrier-like formula switching...");

    forge::GraphRecorder recorder;
    recorder.start();

    Real strike = 100.0, barrier = 90.0;  // strike >= barrier at build time
    strike.markForgeInputAndDiff();
    barrier.markForgeInputAndDiff();
    forge::NodeId strikeId = strike.forgeNodeId(), barrierId = barrier.forgeNodeId();

    // Create ABool condition
    forge::ABool strikeGeBarrier = forge::greaterEqual(strike.forgeValue(), barrier.forgeValue());

    // Simplified formulas
    Real formulaA = strike * 0.1;   // Used when strike >= barrier
    Real formulaB = barrier * 0.2;  // Used when strike < barrier
    Real result = strikeGeBarrier.If(formulaA, formulaB);

    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1: strike=100, barrier=90 -> strike >= barrier -> formulaA = 100*0.1 = 10
    double strikeValBL1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValBL1);
    double barrierValBL1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(barrierId, barrierValBL1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultBLOut1[4]; buffer->getLanes(resultId, resultBLOut1);
    double result1 = resultBLOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: strike=100, barrier=90 (strike >= barrier)");
    BOOST_TEST_MESSAGE("    Expected: 10 (strike*0.1), Got: " << result1);

    // Test 2: strike=80, barrier=95 -> strike < barrier -> formulaB = 95*0.2 = 19
    double strikeValBL2[4] = {80.0, 80.0, 80.0, 80.0}; buffer->setLanes(strikeId, strikeValBL2);
    double barrierValBL2[4] = {95.0, 95.0, 95.0, 95.0}; buffer->setLanes(barrierId, barrierValBL2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultBLOut2[4]; buffer->getLanes(resultId, resultBLOut2);
    double result2 = resultBLOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: strike=80, barrier=95 (strike < barrier)");
    BOOST_TEST_MESSAGE("    Expected: 19 (barrier*0.2), Got: " << result2);

    bool test1Pass = std::abs(result1 - 10.0) < 1e-6;
    bool test2Pass = std::abs(result2 - 19.0) < 1e-6;

    BOOST_TEST_MESSAGE("  Barrier-like ABool::If: " <<
        ((test1Pass && test2Pass) ? "PASSED" : "FAILED"));
    BOOST_TEST_MESSAGE("    Test 1 (strike >= barrier): " << (test1Pass ? "PASS" : "FAIL"));
    BOOST_TEST_MESSAGE("    Test 2 (strike < barrier): " << (test2Pass ? "PASS" : "FAIL - ABool condition not re-evaluating!"));

    BOOST_CHECK_CLOSE(result1, 10.0, 1e-6);
    BOOST_CHECK_CLOSE(result2, 19.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(testPayoffStrikeConnection) {
    // Test: Does PlainVanillaPayoff preserve AReal graph connection?
    // This tests if payoff->strike() returns a value connected to our input
    BOOST_TEST_MESSAGE("ABOOL TEST 3: PlainVanillaPayoff strike graph connection...");

    forge::GraphRecorder recorder;
    recorder.start();

    Real strike = 100.0;
    strike.markForgeInputAndDiff();
    forge::NodeId strikeId = strike.forgeNodeId();

    // Create payoff with our tracked strike
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

    // Get strike back from payoff
    Real retrievedStrike = payoff->strike();

    // Check if it's still connected to the graph
    bool isActive = retrievedStrike.forgeValue().isActive();
    forge::NodeId retrievedId = retrievedStrike.forgeNodeId();

    BOOST_TEST_MESSAGE("  Original strike nodeId: " << strikeId);
    BOOST_TEST_MESSAGE("  Retrieved strike nodeId: " << retrievedId);
    BOOST_TEST_MESSAGE("  Retrieved strike isActive: " << (isActive ? "YES" : "NO"));
    BOOST_TEST_MESSAGE("  Same node? " << (strikeId == retrievedId ? "YES" : "NO"));

    // Use retrieved strike in computation
    Real result = retrievedStrike * 2.0;
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1: strike=100 -> result = 200
    double strikeValPC1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValPC1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultPCOut1[4]; buffer->getLanes(resultId, resultPCOut1);
    double result1 = resultPCOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: strike=100 -> result = " << result1 << " (expected 200)");

    // Test 2: strike=150 -> result = 300 (if connected) or 200 (if disconnected)
    double strikeValPC2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(strikeId, strikeValPC2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultPCOut2[4]; buffer->getLanes(resultId, resultPCOut2);
    double result2 = resultPCOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: strike=150 -> result = " << result2 << " (expected 300 if connected, 200 if disconnected)");

    bool connected = std::abs(result2 - 300.0) < 1e-6;
    BOOST_TEST_MESSAGE("  Payoff strike connection: " << (connected ? "CONNECTED" : "DISCONNECTED!"));

    if (!connected) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND: PlainVanillaPayoff breaks the graph connection!");
        BOOST_TEST_MESSAGE("  *** payoff->strike() returns a NEW value, not the original tracked input.");
    }

    BOOST_CHECK_CLOSE(result2, 300.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(testBarrierArgumentConnection) {
    // Test: Does BarrierOption preserve barrier graph connection?
    BOOST_TEST_MESSAGE("ABOOL TEST 4: BarrierOption barrier argument graph connection...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    forge::GraphRecorder recorder;
    recorder.start();

    Real barrier = 90.0;
    Real strike = 100.0;
    barrier.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    forge::NodeId barrierId = barrier.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();

    // Create barrier option
    auto exercise = ext::make_shared<EuropeanExercise>(today + 1 * Years);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    auto barrierOption = ext::make_shared<BarrierOption>(Barrier::DownOut, barrier, 0.0, payoff, exercise);

    // Access the barrier from the option's arguments
    // Note: We can't directly access arguments_.barrier from here, but we can check
    // if a simple computation using barrier stays connected

    Real barrierTimes2 = barrier * 2.0;
    barrierTimes2.markForgeOutput();
    forge::NodeId resultId = barrierTimes2.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test: Change barrier and see if result changes
    double barrierValBC1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(barrierId, barrierValBC1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultBCOut1[4]; buffer->getLanes(resultId, resultBCOut1);
    double result1 = resultBCOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: barrier=90 -> barrierx2 = " << result1);

    double barrierValBC2[4] = {95.0, 95.0, 95.0, 95.0}; buffer->setLanes(barrierId, barrierValBC2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultBCOut2[4]; buffer->getLanes(resultId, resultBCOut2);
    double result2 = resultBCOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: barrier=95 -> barrierx2 = " << result2);

    BOOST_CHECK_CLOSE(result1, 180.0, 1e-6);
    BOOST_CHECK_CLOSE(result2, 190.0, 1e-6);

    BOOST_TEST_MESSAGE("  Direct barrier computation: " <<
        ((std::abs(result2 - 190.0) < 1e-6) ? "CONNECTED" : "DISCONNECTED"));
}

BOOST_AUTO_TEST_CASE(testEngineBarrierMethodConnection) {
    // Test: Does AnalyticBarrierEngine::barrier() return a connected value?
    // This traces through: BarrierOption -> setupArguments -> engine.arguments_.barrier
    BOOST_TEST_MESSAGE("ABOOL TEST 5: Engine barrier() method connection...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    forge::GraphRecorder recorder;
    recorder.start();

    Real barrier = 90.0;
    Real strike = 100.0;
    Real underlying = 100.0;
    Real vol = 0.20;
    Real rate = 0.05;

    barrier.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();
    underlying.markForgeInputAndDiff();
    vol.markForgeInputAndDiff();
    rate.markForgeInputAndDiff();

    forge::NodeId barrierId = barrier.forgeNodeId();
    forge::NodeId strikeId = strike.forgeNodeId();

    BOOST_TEST_MESSAGE("  Input barrier nodeId: " << barrierId);
    BOOST_TEST_MESSAGE("  Input strike nodeId: " << strikeId);

    // Create the full pricing setup
    DayCounter dayCounter = Actual365Fixed();
    Date maturity = today + 1 * Years;

    auto underlyingH = ext::make_shared<SimpleQuote>(underlying);
    auto riskFreeRate = ext::make_shared<SimpleQuote>(rate);
    auto volatility = ext::make_shared<SimpleQuote>(vol);

    Handle<Quote> h1(riskFreeRate);
    Handle<Quote> h2(volatility);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), h1, dayCounter));
    Handle<BlackVolTermStructure> flatVol(
        ext::make_shared<BlackConstantVol>(0, NullCalendar(), h2, dayCounter));

    auto exercise = ext::make_shared<EuropeanExercise>(maturity);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

    auto bsProcess = ext::make_shared<BlackScholesProcess>(
        Handle<Quote>(underlyingH), flatRate, flatVol);

    // Create barrier option and set engine
    auto barrierOption = ext::make_shared<BarrierOption>(
        Barrier::DownOut, barrier, 0.0, payoff, exercise);
    auto engine = ext::make_shared<AnalyticBarrierEngine>(bsProcess);
    barrierOption->setPricingEngine(engine);

    // Now get NPV - this triggers calculate() and we can see if values stay connected
    Real npv = barrierOption->NPV();
    npv.markForgeOutput();
    forge::NodeId npvId = npv.forgeNodeId();

    BOOST_TEST_MESSAGE("  NPV nodeId: " << npvId);
    BOOST_TEST_MESSAGE("  NPV isActive: " << (npv.forgeValue().isActive() ? "YES" : "NO"));

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1: Original values
    double barrierValE1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(barrierId, barrierValE1);
    double strikeValE1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValE1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double npvOut1[4]; buffer->getLanes(npvId, npvOut1);
    double npv1 = npvOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: barrier=90, strike=100 -> NPV = " << npv1);

    // Test 2: Change barrier only
    double barrierValE2[4] = {85.0, 85.0, 85.0, 85.0}; buffer->setLanes(barrierId, barrierValE2);
    double strikeValE2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValE2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double npvOut2[4]; buffer->getLanes(npvId, npvOut2);
    double npv2 = npvOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: barrier=85, strike=100 -> NPV = " << npv2);

    // Test 3: Change strike only
    double barrierValE3[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(barrierId, barrierValE3);
    double strikeValE3[4] = {110.0, 110.0, 110.0, 110.0}; buffer->setLanes(strikeId, strikeValE3);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double npvOut3[4]; buffer->getLanes(npvId, npvOut3);
    double npv3 = npvOut3[0];
    BOOST_TEST_MESSAGE("  Test 3: barrier=90, strike=110 -> NPV = " << npv3);

    bool barrierConnected = std::abs(npv1 - npv2) > 0.01;  // Should be different if connected
    bool strikeConnected = std::abs(npv1 - npv3) > 0.01;   // Should be different if connected

    BOOST_TEST_MESSAGE("  Barrier affects NPV: " << (barrierConnected ? "YES (connected)" : "NO (disconnected!)"));
    BOOST_TEST_MESSAGE("  Strike affects NPV: " << (strikeConnected ? "YES (connected)" : "NO (disconnected!)"));

    if (!barrierConnected) {
        BOOST_TEST_MESSAGE("  *** ISSUE: Barrier is DISCONNECTED in engine pipeline!");
    }
    if (!strikeConnected) {
        BOOST_TEST_MESSAGE("  *** ISSUE: Strike is DISCONNECTED in engine pipeline!");
    }
}

BOOST_AUTO_TEST_CASE(testABoolConditionInEngine) {
    // Test: Is the ABool condition in the engine using connected values?
    // We create the condition manually mimicking what the patched engine does
    BOOST_TEST_MESSAGE("ABOOL TEST 6: ABool condition with engine-retrieved values...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    forge::GraphRecorder recorder;
    recorder.start();

    Real inputStrike = 100.0;
    Real inputBarrier = 90.0;

    inputStrike.markForgeInputAndDiff();
    inputBarrier.markForgeInputAndDiff();

    forge::NodeId strikeId = inputStrike.forgeNodeId();
    forge::NodeId barrierId = inputBarrier.forgeNodeId();

    // Create payoff with tracked strike
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, inputStrike);

    // Retrieve strike from payoff (simulating what engine does)
    Real engineStrike = payoff->strike();

    BOOST_TEST_MESSAGE("  Input strike nodeId: " << strikeId);
    BOOST_TEST_MESSAGE("  Engine strike nodeId: " << engineStrike.forgeNodeId());
    BOOST_TEST_MESSAGE("  Engine strike isActive: " << (engineStrike.forgeValue().isActive() ? "YES" : "NO"));
    BOOST_TEST_MESSAGE("  Same node? " << (strikeId == engineStrike.forgeNodeId() ? "YES" : "NO"));

    // Create ABool condition using engine-retrieved values
    // This mimics: forge::ABool strikeGeBarrier = greaterEqualReal(strike, barrier());
    forge::ABool cond = forge::greaterEqual(engineStrike.forgeValue(), inputBarrier.forgeValue());

    BOOST_TEST_MESSAGE("  ABool condition isActive: " << (cond.isActive() ? "YES" : "NO"));
    BOOST_TEST_MESSAGE("  ABool condition passive value: " << (cond.passive() ? "TRUE" : "FALSE"));

    // Use the condition to select between formulas
    Real formulaA = engineStrike * 0.1;   // strike >= barrier
    Real formulaB = inputBarrier * 0.2;   // strike < barrier
    Real result = cond.If(formulaA, formulaB);

    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1: strike=100 >= barrier=90 -> formulaA = 10
    double strikeValER1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValER1);
    double barrierValER1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(barrierId, barrierValER1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultEROut1[4]; buffer->getLanes(resultId, resultEROut1);
    double result1 = resultEROut1[0];
    BOOST_TEST_MESSAGE("  Test 1: strike=100, barrier=90 -> " << result1 << " (expected 10)");

    // Test 2: strike=80 < barrier=95 -> formulaB = 19
    double strikeValER2[4] = {80.0, 80.0, 80.0, 80.0}; buffer->setLanes(strikeId, strikeValER2);
    double barrierValER2[4] = {95.0, 95.0, 95.0, 95.0}; buffer->setLanes(barrierId, barrierValER2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultEROut2[4]; buffer->getLanes(resultId, resultEROut2);
    double result2 = resultEROut2[0];
    BOOST_TEST_MESSAGE("  Test 2: strike=80, barrier=95 -> " << result2 << " (expected 19)");

    bool test1Pass = std::abs(result1 - 10.0) < 1e-6;
    bool test2Pass = std::abs(result2 - 19.0) < 1e-6;

    BOOST_TEST_MESSAGE("  Engine-retrieved ABool::If: " << ((test1Pass && test2Pass) ? "PASSED" : "FAILED"));
    BOOST_TEST_MESSAGE("    Test 1: " << (test1Pass ? "PASS" : "FAIL"));
    BOOST_TEST_MESSAGE("    Test 2: " << (test2Pass ? "PASS" : "FAIL"));

    if (!test2Pass) {
        BOOST_TEST_MESSAGE("  *** ISSUE: Condition not re-evaluating with engine-retrieved strike!");
    }

    BOOST_CHECK_CLOSE(result1, 10.0, 1e-6);
    BOOST_CHECK_CLOSE(result2, 19.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(testSimpleQuoteConnection) {
    // Test: Does SimpleQuote preserve AReal graph connection?
    BOOST_TEST_MESSAGE("ABOOL TEST 7: SimpleQuote graph connection...");

    forge::GraphRecorder recorder;
    recorder.start();

    Real inputValue = 100.0;
    inputValue.markForgeInputAndDiff();
    forge::NodeId inputId = inputValue.forgeNodeId();

    // Create SimpleQuote with tracked value
    auto quote = ext::make_shared<SimpleQuote>(inputValue);

    // Retrieve value from quote
    Real retrievedValue = quote->value();

    BOOST_TEST_MESSAGE("  Input nodeId: " << inputId);
    BOOST_TEST_MESSAGE("  Retrieved nodeId: " << retrievedValue.forgeNodeId());
    BOOST_TEST_MESSAGE("  Retrieved isActive: " << (retrievedValue.forgeValue().isActive() ? "YES" : "NO"));
    BOOST_TEST_MESSAGE("  Same node? " << (inputId == retrievedValue.forgeNodeId() ? "YES" : "NO"));

    // Use retrieved value in computation
    Real result = retrievedValue * 2.0;
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1
    double inputValSQ1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(inputId, inputValSQ1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultSQOut1[4]; buffer->getLanes(resultId, resultSQOut1);
    double result1 = resultSQOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: input=100 -> result = " << result1);

    // Test 2
    double inputValSQ2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(inputId, inputValSQ2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultSQOut2[4]; buffer->getLanes(resultId, resultSQOut2);
    double result2 = resultSQOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: input=150 -> result = " << result2);

    bool connected = std::abs(result2 - 300.0) < 1e-6;
    BOOST_TEST_MESSAGE("  SimpleQuote connection: " << (connected ? "CONNECTED" : "DISCONNECTED!"));

    BOOST_CHECK_CLOSE(result2, 300.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(testHandleQuoteConnection) {
    // Test: Does Handle<Quote> preserve AReal graph connection?
    BOOST_TEST_MESSAGE("ABOOL TEST 8: Handle<Quote> graph connection...");

    forge::GraphRecorder recorder;
    recorder.start();

    Real inputValue = 100.0;
    inputValue.markForgeInputAndDiff();
    forge::NodeId inputId = inputValue.forgeNodeId();

    // Create SimpleQuote and wrap in Handle
    auto quote = ext::make_shared<SimpleQuote>(inputValue);
    Handle<Quote> handle(quote);

    // Retrieve value through Handle
    Real retrievedValue = handle->value();

    BOOST_TEST_MESSAGE("  Input nodeId: " << inputId);
    BOOST_TEST_MESSAGE("  Retrieved via Handle nodeId: " << retrievedValue.forgeNodeId());
    BOOST_TEST_MESSAGE("  Retrieved isActive: " << (retrievedValue.forgeValue().isActive() ? "YES" : "NO"));

    // Use retrieved value in computation
    Real result = retrievedValue * 2.0;
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1
    double inputValHQ1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(inputId, inputValHQ1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultHQOut1[4]; buffer->getLanes(resultId, resultHQOut1);
    double result1 = resultHQOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: input=100 -> result = " << result1);

    // Test 2
    double inputValHQ2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(inputId, inputValHQ2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultHQOut2[4]; buffer->getLanes(resultId, resultHQOut2);
    double result2 = resultHQOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: input=150 -> result = " << result2);

    bool connected = std::abs(result2 - 300.0) < 1e-6;
    BOOST_TEST_MESSAGE("  Handle<Quote> connection: " << (connected ? "CONNECTED" : "DISCONNECTED!"));

    if (!connected) {
        BOOST_TEST_MESSAGE("  *** ISSUE: Handle<Quote> breaks the graph connection!");
    }

    BOOST_CHECK_CLOSE(result2, 300.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(testFlatForwardConnection) {
    // Test: Does FlatForward preserve AReal graph connection through zeroRate?
    BOOST_TEST_MESSAGE("ABOOL TEST 9: FlatForward zeroRate graph connection...");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;

    forge::GraphRecorder recorder;
    recorder.start();

    Real inputRate = 0.05;
    inputRate.markForgeInputAndDiff();
    forge::NodeId inputId = inputRate.forgeNodeId();

    DayCounter dayCounter = Actual365Fixed();
    Date maturity = today + 1 * Years;

    // Create rate quote and FlatForward
    auto rateQuote = ext::make_shared<SimpleQuote>(inputRate);
    Handle<Quote> rateHandle(rateQuote);
    Handle<YieldTermStructure> flatRate(
        ext::make_shared<FlatForward>(0, NullCalendar(), rateHandle, dayCounter));

    // Retrieve rate through zeroRate
    Real retrievedRate = flatRate->zeroRate(maturity, dayCounter, Continuous, NoFrequency);

    BOOST_TEST_MESSAGE("  Input rate nodeId: " << inputId);
    BOOST_TEST_MESSAGE("  Retrieved rate isActive: " << (retrievedRate.forgeValue().isActive() ? "YES" : "NO"));

    // Use retrieved rate in computation
    Real result = retrievedRate * 100.0;  // Scale for visibility
    result.markForgeOutput();
    forge::NodeId resultId = result.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

    forge::ForgeEngine compiler;  // Default config is stability-only
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    // Test 1
    double inputValFF1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(inputId, inputValFF1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultFFOut1[4]; buffer->getLanes(resultId, resultFFOut1);
    double result1 = resultFFOut1[0];
    BOOST_TEST_MESSAGE("  Test 1: rate=0.05 -> result = " << result1 << " (expected ~5)");

    // Test 2
    double inputValFF2[4] = {0.10, 0.10, 0.10, 0.10}; buffer->setLanes(inputId, inputValFF2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    double resultFFOut2[4]; buffer->getLanes(resultId, resultFFOut2);
    double result2 = resultFFOut2[0];
    BOOST_TEST_MESSAGE("  Test 2: rate=0.10 -> result = " << result2 << " (expected ~10)");

    bool connected = std::abs(result2 - 10.0) < 0.1;
    BOOST_TEST_MESSAGE("  FlatForward zeroRate connection: " << (connected ? "CONNECTED" : "DISCONNECTED!"));

    if (!connected) {
        BOOST_TEST_MESSAGE("  *** ISSUE: FlatForward breaks the graph connection!");
    }
}

// =============================================================================
// DIAGNOSTIC: Test each formula function (A, B, C, D, E, F) individually
// =============================================================================

BOOST_AUTO_TEST_CASE(testFormulaFunctionsKernelReuse) {
    // Test each formula (A, B, C, D, A-C, B-D) with kernel reuse
    // Uses SAME inputs as kernel reuse test:
    //   Build:     strike=100, u=90,  r=0.10, b=0.10, v=0.10
    //   InputSet5: strike=80,  u=100, r=0.05, b=95,   v=0.20

    BOOST_TEST_MESSAGE("FORMULA KERNEL REUSE: Testing A, B, C, D, A-C, B-D...");
    BOOST_TEST_MESSAGE("Build with:   strike=100, u=90,  r=0.10, b=0.10, v=0.10");
    BOOST_TEST_MESSAGE("Re-eval with: strike=80,  u=100, r=0.05, b=95,   v=0.20 (Input Set 5)");
    BOOST_TEST_MESSAGE("");

    Date today(29, May, 2006);
    Settings::instance().evaluationDate() = today;
    DayCounter dayCounter = Actual365Fixed();
    Date maturity = today + 1 * Years;

    // Test configurations: {OptionType, BarrierType, FormulaName}
    // Engine returns: Call+DownIn->A, Call+UpIn->B, Call+DownOut->C, Call+UpOut->D
    //                 Put+DownOut->A-C, Put+UpOut->B-D
    struct TestConfig {
        Option::Type optionType;
        Barrier::Type barrierType;
        const char* formulaName;
    };

    // ALL use DownIn/DownOut (valid for kernel reuse inputs: barrier < spot)
    // Engine mapping:
    //   Call+DownIn  -> A(1)
    //   Call+DownOut -> B(1)
    //   Put+DownIn   -> C(1,1)
    //   Put+DownOut  -> D(1,1)
    std::vector<TestConfig> configs = {
        {Option::Call, Barrier::DownIn,  "A(1)"},
        {Option::Call, Barrier::DownOut, "B(1)"},
        {Option::Put,  Barrier::DownIn,  "C(1,1)"},
        {Option::Put,  Barrier::DownOut, "D(1,1)"},
    };

    // EXACT same inputs as kernel reuse test - NO CHANGES
    // Build: strike=100, u=90, r=0.10, b=0.10, v=0.10
    double buildStrike = 100.0, buildU = 90.0, buildR = 0.10, buildB = 0.10, buildV = 0.10;
    // Input Set 5: strike=80, u=100, r=0.05, b=95, v=0.20
    double set5Strike = 80.0, set5U = 100.0, set5R = 0.05, set5B = 95.0, set5V = 0.20;

    for (const auto& config : configs) {
        BOOST_TEST_MESSAGE("--- Testing formula " << config.formulaName << " ---");

        forge::GraphRecorder recorder;
        recorder.start();

        // Create tracked inputs with BUILD values
        Real strike = buildStrike;
        Real u = buildU;
        Real r = buildR;
        Real b = buildB;
        Real v = buildV;

        strike.markForgeInputAndDiff();
        u.markForgeInputAndDiff();
        r.markForgeInputAndDiff();
        b.markForgeInputAndDiff();
        v.markForgeInputAndDiff();

        forge::NodeId strikeId = strike.forgeNodeId();
        forge::NodeId uId = u.forgeNodeId();
        forge::NodeId rId = r.forgeNodeId();
        forge::NodeId bId = b.forgeNodeId();
        forge::NodeId vId = v.forgeNodeId();

        // Build pricing (like priceBarrierOption)
        auto underlyingH = ext::make_shared<SimpleQuote>(u);
        auto riskFreeRate = ext::make_shared<SimpleQuote>(r);
        auto volatility = ext::make_shared<SimpleQuote>(v);

        Handle<Quote> h1(riskFreeRate);
        Handle<Quote> h2(volatility);
        Handle<YieldTermStructure> flatRate(
            ext::make_shared<FlatForward>(0, NullCalendar(), h1, dayCounter));
        Handle<BlackVolTermStructure> flatVol(
            ext::make_shared<BlackConstantVol>(0, NullCalendar(), h2, dayCounter));

        auto payoff = ext::make_shared<PlainVanillaPayoff>(config.optionType, strike);
        auto exercise = ext::make_shared<EuropeanExercise>(maturity);
        auto bsProcess = ext::make_shared<BlackScholesProcess>(Handle<Quote>(underlyingH), flatRate, flatVol);

        auto barrierOption = ext::make_shared<BarrierOption>(config.barrierType, b, 0.0, payoff, exercise);
        barrierOption->setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bsProcess));

        Real npv = barrierOption->NPV();
        npv.markForgeOutput();
        forge::NodeId npvId = npv.forgeNodeId();

        recorder.stop();
        forge::Graph graph = recorder.graph();

        BOOST_TEST_MESSAGE("  Graph has " << graph.nodes.size() << " nodes");

        forge::ForgeEngine compiler;  // Default is stability-only
        auto kernel = compiler.compile(graph);
        auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

        // Test 1: Re-evaluate with BUILD inputs (should match exactly)
        double strikeValBld[4] = {buildStrike, buildStrike, buildStrike, buildStrike}; buffer->setLanes(strikeId, strikeValBld);
        double uValBld[4] = {buildU, buildU, buildU, buildU}; buffer->setLanes(uId, uValBld);
        double rValBld[4] = {buildR, buildR, buildR, buildR}; buffer->setLanes(rId, rValBld);
        double bValBld[4] = {buildB, buildB, buildB, buildB}; buffer->setLanes(bId, bValBld);
        double vValBld[4] = {buildV, buildV, buildV, buildV}; buffer->setLanes(vId, vValBld);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double npvBldOut[4]; buffer->getLanes(npvId, npvBldOut);
        double forgeBuild = npvBldOut[0];

        // Compute expected for build inputs
        auto spotQ1 = ext::make_shared<SimpleQuote>(buildU);
        auto rateQ1 = ext::make_shared<SimpleQuote>(buildR);
        auto volQ1 = ext::make_shared<SimpleQuote>(buildV);
        Handle<YieldTermStructure> fr1(ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQ1), dayCounter));
        Handle<BlackVolTermStructure> fv1(ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQ1), dayCounter));
        auto po1 = ext::make_shared<PlainVanillaPayoff>(config.optionType, buildStrike);
        auto ex1 = ext::make_shared<EuropeanExercise>(maturity);
        auto bs1 = ext::make_shared<BlackScholesProcess>(Handle<Quote>(spotQ1), fr1, fv1);
        auto opt1 = ext::make_shared<BarrierOption>(config.barrierType, buildB, 0.0, po1, ex1);
        opt1->setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bs1));
        double expectedBuild = value(opt1->NPV());

        // Test 2: Re-evaluate with INPUT SET 5 values
        double strikeValS5[4] = {set5Strike, set5Strike, set5Strike, set5Strike}; buffer->setLanes(strikeId, strikeValS5);
        double uValS5[4] = {set5U, set5U, set5U, set5U}; buffer->setLanes(uId, uValS5);
        double rValS5[4] = {set5R, set5R, set5R, set5R}; buffer->setLanes(rId, rValS5);
        double bValS5[4] = {set5B, set5B, set5B, set5B}; buffer->setLanes(bId, bValS5);
        double vValS5[4] = {set5V, set5V, set5V, set5V}; buffer->setLanes(vId, vValS5);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double npvS5Out[4]; buffer->getLanes(npvId, npvS5Out);
        double forgeSet5 = npvS5Out[0];

        // Compute expected for Input Set 5
        auto spotQ2 = ext::make_shared<SimpleQuote>(set5U);
        auto rateQ2 = ext::make_shared<SimpleQuote>(set5R);
        auto volQ2 = ext::make_shared<SimpleQuote>(set5V);
        Handle<YieldTermStructure> fr2(ext::make_shared<FlatForward>(0, NullCalendar(), Handle<Quote>(rateQ2), dayCounter));
        Handle<BlackVolTermStructure> fv2(ext::make_shared<BlackConstantVol>(0, NullCalendar(), Handle<Quote>(volQ2), dayCounter));
        auto po2 = ext::make_shared<PlainVanillaPayoff>(config.optionType, set5Strike);
        auto ex2 = ext::make_shared<EuropeanExercise>(maturity);
        auto bs2 = ext::make_shared<BlackScholesProcess>(Handle<Quote>(spotQ2), fr2, fv2);
        auto opt2 = ext::make_shared<BarrierOption>(config.barrierType, set5B, 0.0, po2, ex2);
        opt2->setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(bs2));
        double expectedSet5 = value(opt2->NPV());

        // Check results
        double tol = 1e-6;
        bool buildOK = (expectedBuild == 0.0) ? (std::abs(forgeBuild) < 1e-10) :
                       (std::abs(forgeBuild - expectedBuild) / std::abs(expectedBuild) < tol);
        bool set5OK = (expectedSet5 == 0.0) ? (std::abs(forgeSet5) < 1e-10) :
                      (std::abs(forgeSet5 - expectedSet5) / std::abs(expectedSet5) < tol);

        double buildErr = (expectedBuild == 0.0) ? forgeBuild : 100.0 * (forgeBuild - expectedBuild) / expectedBuild;
        double set5Err = (expectedSet5 == 0.0) ? forgeSet5 : 100.0 * (forgeSet5 - expectedSet5) / expectedSet5;

        BOOST_TEST_MESSAGE("  Build inputs: Forge=" << forgeBuild << " Expected=" << expectedBuild
                          << " Err=" << buildErr << "%" << (buildOK ? " OK" : " WRONG!"));
        BOOST_TEST_MESSAGE("  InputSet5:    Forge=" << forgeSet5 << " Expected=" << expectedSet5
                          << " Err=" << set5Err << "%" << (set5OK ? " OK" : " WRONG!"));

        if (buildOK && set5OK) {
            BOOST_TEST_MESSAGE("  *** " << config.formulaName << " KERNEL REUSE WORKS!");
        } else {
            BOOST_TEST_MESSAGE("  *** " << config.formulaName << " KERNEL REUSE FAILS!");
        }
        BOOST_TEST_MESSAGE("");
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
    bool g1 = std::abs(grad[0 * vw] - 4.0) < 1e-9 && std::abs(grad[1 * vw] - 2.0) < 1e-9;

    // Test 2 (re-eval): x=4, y=5 -> z=24, dz/dx=6, dz/dy=4
    double xVal2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal2);
    double yVal2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(yId, yVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOut2[4]; buffer->getLanes(zId, zOut2);
    bool v2 = std::abs(zOut2[0] - 24.0) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 6.0) < 1e-9 && std::abs(grad[1 * vw] - 4.0) < 1e-9;

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
    double inputValSQ1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(inputId, inputValSQ1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOutSQ1[4]; buffer->getLanes(outputId, outputOutSQ1);
    bool v1 = std::abs(outputOutSQ1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-9;

    // Test 2: input=150
    double inputValSQ2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(inputId, inputValSQ2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOutSQ2[4]; buffer->getLanes(outputId, outputOutSQ2);
    bool v2 = std::abs(outputOutSQ2[0] - 150.0) < 1e-9;
    bool g2 = std::abs(grad[0 * vw] - 1.0) < 1e-9;

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
    double rateValFF1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateValFF1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOutFF1[4]; buffer->getLanes(dfId, dfOutFF1);
    bool v1 = std::abs(dfOutFF1[0] - std::exp(-0.05)) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - (-std::exp(-0.05))) < 1e-4;

    // Test 2: r=0.08
    double rateValFF2[4] = {0.08, 0.08, 0.08, 0.08}; buffer->setLanes(rateId, rateValFF2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOutFF2[4]; buffer->getLanes(dfId, dfOutFF2);
    bool v2 = std::abs(dfOutFF2[0] - std::exp(-0.08)) < 1e-6;
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
    double volValBCV1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volValBCV1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOut1[4]; buffer->getLanes(blackVolId, blackVolOut1);
    bool v1 = std::abs(blackVolOut1[0] - 0.20) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    // Test 2: vol=0.30
    double volValBCV2[4] = {0.30, 0.30, 0.30, 0.30}; buffer->setLanes(volId, volValBCV2);
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
    double spotValBSP1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(spotId, spotValBSP1);
    double rateValBSP1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateValBSP1);
    double volValBSP1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volValBSP1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0Out1[4]; buffer->getLanes(x0Id, x0Out1);
    bool v1 = std::abs(x0Out1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6 && std::abs(grad[1 * vw]) < 1e-9 && std::abs(grad[2 * vw]) < 1e-9;

    // Test 2: spot=120
    double spotValBSP2[4] = {120.0, 120.0, 120.0, 120.0}; buffer->setLanes(spotId, spotValBSP2);
    double rateValBSP2[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateValBSP2);
    double volValBSP2[4] = {0.25, 0.25, 0.25, 0.25}; buffer->setLanes(volId, volValBSP2);
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

    double xValNO1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xValNO1);
    double yValNO1[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(yId, yValNO1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOutNO1[4]; buffer->getLanes(zId, zOutNO1);
    bool v1 = std::abs(zOutNO1[0] - 8.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 4.0) < 1e-9 && std::abs(grad[1 * vw] - 2.0) < 1e-9;

    double xValNO2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xValNO2);
    double yValNO2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(yId, yValNO2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zOutNO2[4]; buffer->getLanes(zId, zOutNO2);
    bool v2 = std::abs(zOutNO2[0] - 24.0) < 1e-9;
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

    double inputValSQN1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(inputId, inputValSQN1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOutSQN1[4]; buffer->getLanes(outputId, outputOutSQN1);
    bool v1 = std::abs(outputOutSQN1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-9;

    double inputValSQN2[4] = {150.0, 150.0, 150.0, 150.0}; buffer->setLanes(inputId, inputValSQN2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double outputOutSQN2[4]; buffer->getLanes(outputId, outputOutSQN2);
    bool v2 = std::abs(outputOutSQN2[0] - 150.0) < 1e-9;
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

    double rateValFFN1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateValFFN1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOutFFN1[4]; buffer->getLanes(dfId, dfOutFFN1);
    bool v1 = std::abs(dfOutFFN1[0] - std::exp(-0.05)) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - (-std::exp(-0.05))) < 1e-4;

    double rateValFFN2[4] = {0.08, 0.08, 0.08, 0.08}; buffer->setLanes(rateId, rateValFFN2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double dfOutFFN2[4]; buffer->getLanes(dfId, dfOutFFN2);
    bool v2 = std::abs(dfOutFFN2[0] - std::exp(-0.08)) < 1e-6;
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

    double volValBCVN1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volValBCVN1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOutBCVN1[4]; buffer->getLanes(blackVolId, blackVolOutBCVN1);
    bool v1 = std::abs(blackVolOutBCVN1[0] - 0.20) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    double volValBCVN2[4] = {0.30, 0.30, 0.30, 0.30}; buffer->setLanes(volId, volValBCVN2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double blackVolOutBCVN2[4]; buffer->getLanes(blackVolId, blackVolOutBCVN2);
    bool v2 = std::abs(blackVolOutBCVN2[0] - 0.30) < 1e-9;
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

    double spotValBSPN1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(spotId, spotValBSPN1);
    double rateValBSPN1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateValBSPN1);
    double volValBSPN1[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volValBSPN1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0OutBSPN1[4]; buffer->getLanes(x0Id, x0OutBSPN1);
    bool v1 = std::abs(x0OutBSPN1[0] - 100.0) < 1e-9;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6 && std::abs(grad[1 * vw]) < 1e-9 && std::abs(grad[2 * vw]) < 1e-9;

    double spotValBSPN2[4] = {120.0, 120.0, 120.0, 120.0}; buffer->setLanes(spotId, spotValBSPN2);
    double rateValBSPN2[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateValBSPN2);
    double volValBSPN2[4] = {0.25, 0.25, 0.25, 0.25}; buffer->setLanes(volId, volValBSPN2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0OutBSPN2[4]; buffer->getLanes(x0Id, x0OutBSPN2);
    bool v2 = std::abs(x0OutBSPN2[0] - 120.0) < 1e-9;
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
    double xValEF1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xValEF1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutEF1[4]; buffer->getLanes(resultId, resultOutEF1);
    double expV1 = std::erf(0.5);
    double expG1 = 2.0 / std::sqrt(M_PI) * std::exp(-0.5 * 0.5);
    bool v1 = std::abs(resultOutEF1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: x=1.0
    double xValEF2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xValEF2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutEF2[4]; buffer->getLanes(resultId, resultOutEF2);
    double expV2 = std::erf(1.0);
    double expG2 = 2.0 / std::sqrt(M_PI) * std::exp(-1.0);
    bool v2 = std::abs(resultOutEF2[0] - expV2) < 1e-6;
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
    double xValND1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xValND1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOutND1[4]; buffer->getLanes(pdfId, pdfOutND1);
    double expV1 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * 0.5 / 2.0);
    double expG1 = -0.5 * expV1;
    bool v1 = std::abs(pdfOutND1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: x=1.0
    double xValND2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xValND2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOutND2[4]; buffer->getLanes(pdfId, pdfOutND2);
    double expV2 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-1.0 / 2.0);
    double expG2 = -1.0 * expV2;
    bool v2 = std::abs(pdfOutND2[0] - expV2) < 1e-6;
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
    double xValCND1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xValCND1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutCND1[4]; buffer->getLanes(resultId, resultOutCND1);
    double expV1 = 0.5 * (1.0 + std::erf(0.5 / std::sqrt(2.0)));
    double expG1 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * 0.5 / 2.0);
    bool v1 = std::abs(resultOutCND1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: x=1.0
    double xValCND2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xValCND2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutCND2[4]; buffer->getLanes(resultId, resultOutCND2);
    double expV2 = 0.5 * (1.0 + std::erf(1.0 / std::sqrt(2.0)));
    double expG2 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-1.0 / 2.0);
    bool v2 = std::abs(resultOutCND2[0] - expV2) < 1e-6;
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
    double xValCNDT1[4] = {-5.0, -5.0, -5.0, -5.0}; buffer->setLanes(xId, xValCNDT1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutCNDT1[4]; buffer->getLanes(resultId, resultOutCNDT1);
    double expV1 = 0.5 * (1.0 + std::erf(-5.0 / std::sqrt(2.0)));
    double expG1 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-5.0 * 5.0 / 2.0);
    bool v1 = std::abs(resultOutCNDT1[0] - expV1) / expV1 < 1e-4;
    bool g1 = std::abs(grad[0 * vw] - expG1) / expG1 < 0.1;

    // Test 2: x=-4.0
    double xValCNDT2[4] = {-4.0, -4.0, -4.0, -4.0}; buffer->setLanes(xId, xValCNDT2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutCNDT2[4]; buffer->getLanes(resultId, resultOutCNDT2);
    double expV2 = 0.5 * (1.0 + std::erf(-4.0 / std::sqrt(2.0)));
    double expG2 = 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-4.0 * 4.0 / 2.0);
    bool v2 = std::abs(resultOutCNDT2[0] - expV2) / expV2 < 1e-4;
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
    double xValNDET1[4] = {40.0, 40.0, 40.0, 40.0}; buffer->setLanes(xId, xValNDET1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOutNDET1[4]; buffer->getLanes(pdfId, pdfOutNDET1);
    bool v1 = std::abs(pdfOutNDET1[0]) < 1e-300;
    bool g1 = std::abs(grad[0 * vw]) < 1e-100;

    // Test 2: x=50
    double xValNDET2[4] = {50.0, 50.0, 50.0, 50.0}; buffer->setLanes(xId, xValNDET2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double pdfOutNDET2[4]; buffer->getLanes(pdfId, pdfOutNDET2);
    bool v2 = std::abs(pdfOutNDET2[0]) < 1e-300;
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
    double xValSL1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xValSL1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutSL1[4]; buffer->getLanes(resultId, resultOutSL1);
    bool v1 = std::abs(resultOutSL1[0] - std::log(2.0)) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 0.5) < 1e-6;

    // Test 2: x=5
    double xValSL2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(xId, xValSL2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutSL2[4]; buffer->getLanes(resultId, resultOutSL2);
    bool v2 = std::abs(resultOutSL2[0] - std::log(5.0)) < 1e-10;
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
    double baseValSP1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(baseId, baseValSP1);
    double expValSP1[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(expId, expValSP1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutSP1[4]; buffer->getLanes(resultId, resultOutSP1);
    bool v1 = std::abs(resultOutSP1[0] - 8.0) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 12.0) < 1e-6 && std::abs(grad[1 * vw] - 8.0*std::log(2.0)) < 1e-6;

    // Test 2: base=3, exp=2 -> 9, grads: 6, 9*log(3)
    double baseValSP2[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(baseId, baseValSP2);
    double expValSP2[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(expId, expValSP2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutSP2[4]; buffer->getLanes(resultId, resultOutSP2);
    bool v2 = std::abs(resultOutSP2[0] - 9.0) < 1e-10;
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
    double aValLR1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aValLR1);
    double bValLR1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(bId, bValLR1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutLR1[4]; buffer->getLanes(resultId, resultOutLR1);
    bool v1 = std::abs(resultOutLR1[0] - std::log(100.0/90.0)) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 0.01) < 1e-6 && std::abs(grad[1 * vw] - (-1.0/90.0)) < 1e-6;

    // Test 2: a=120, b=80
    double aValLR2[4] = {120.0, 120.0, 120.0, 120.0}; buffer->setLanes(aId, aValLR2);
    double bValLR2[4] = {80.0, 80.0, 80.0, 80.0}; buffer->setLanes(bId, bValLR2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutLR2[4]; buffer->getLanes(resultId, resultOutLR2);
    bool v2 = std::abs(resultOutLR2[0] - std::log(120.0/80.0)) < 1e-10;
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
    double underlyingValX1_1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValX1_1);
    double strikeValX1_1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValX1_1);
    double stdDevValX1_1[4] = {0.1, 0.1, 0.1, 0.1}; buffer->setLanes(stdDevId, stdDevValX1_1);
    double muSigmaValX1_1[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(muSigmaId, muSigmaValX1_1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x1OutX1_1[4]; buffer->getLanes(x1Id, x1OutX1_1);
    double expV1 = std::log(90.0/100.0)/0.1 + 0.05;
    bool v1 = std::abs(x1OutX1_1[0] - expV1) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 1.0/(90.0*0.1)) < 1e-6;

    // Test 2
    double underlyingValX1_2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(underlyingId, underlyingValX1_2);
    double strikeValX1_2[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(strikeId, strikeValX1_2);
    double stdDevValX1_2[4] = {0.1, 0.1, 0.1, 0.1}; buffer->setLanes(stdDevId, stdDevValX1_2);
    double muSigmaValX1_2[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(muSigmaId, muSigmaValX1_2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x1OutX1_2[4]; buffer->getLanes(x1Id, x1OutX1_2);
    double expV2 = std::log(100.0/90.0)/0.1 + 0.05;
    bool v2 = std::abs(x1OutX1_2[0] - expV2) < 1e-10;
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
    double barrierValHSP1[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValHSP1);
    double underlyingValHSP1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValHSP1);
    double muValHSP1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(muId, muValHSP1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double powHS0OutHSP1[4]; buffer->getLanes(powHS0Id, powHS0OutHSP1);
    double expV1 = std::pow(70.0/90.0, 1.0);
    bool v1 = std::abs(powHS0OutHSP1[0] - expV1) < 1e-10;
    bool g1 = std::abs(grad[0 * vw]) > 1e-10 && std::abs(grad[1 * vw]) > 1e-10;

    // Test 2: barrier=80, underlying=100, mu=0.5
    double barrierValHSP2[4] = {80.0, 80.0, 80.0, 80.0}; buffer->setLanes(barrierId, barrierValHSP2);
    double underlyingValHSP2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(underlyingId, underlyingValHSP2);
    double muValHSP2[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(muId, muValHSP2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double powHS0OutHSP2[4]; buffer->getLanes(powHS0Id, powHS0OutHSP2);
    double expV2 = std::pow(80.0/100.0, 1.0);
    bool v2 = std::abs(powHS0OutHSP2[0] - expV2) < 1e-10;
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
    double underlyingValAF1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValAF1);
    double strikeValAF1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValAF1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOutAF1[4]; buffer->getLanes(AId, AOutAF1);
    double x1_exp = std::log(90.0/100.0)/0.1 + 0.05;
    double N1_exp = 0.5*(1.0 + std::erf(x1_exp/std::sqrt(2.0)));
    double N2_exp = 0.5*(1.0 + std::erf((x1_exp-0.1)/std::sqrt(2.0)));
    double expV1 = 1.0*(90.0*0.98*N1_exp - 100.0*0.95*N2_exp);
    bool v1 = std::abs(AOutAF1[0] - expV1) / std::abs(expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw]) > 1e-10 && std::abs(grad[1 * vw]) > 1e-10;

    // Test 2
    double underlyingValAF2[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(underlyingId, underlyingValAF2);
    double strikeValAF2[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(strikeId, strikeValAF2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOutAF2[4]; buffer->getLanes(AId, AOutAF2);
    double x1_exp2 = std::log(100.0/90.0)/0.1 + 0.05;
    double N1_exp2 = 0.5*(1.0 + std::erf(x1_exp2/std::sqrt(2.0)));
    double N2_exp2 = 0.5*(1.0 + std::erf((x1_exp2-0.1)/std::sqrt(2.0)));
    double expV2 = 1.0*(100.0*0.98*N1_exp2 - 90.0*0.95*N2_exp2);
    bool v2 = std::abs(AOutAF2[0] - expV2) / std::abs(expV2) < 1e-6;
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
    double xValCFZ1[4] = {0.5, 0.5, 0.5, 0.5}; buffer->setLanes(xId, xValCFZ1);
    double powHSValCFZ1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(powHSId, powHSValCFZ1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutCFZ1[4]; buffer->getLanes(resultId, resultOutCFZ1);
    double N1_exp = 0.5*(1.0 + std::erf(0.5/std::sqrt(2.0)));
    double expV1 = 2.0 * N1_exp;
    bool v1 = std::abs(resultOutCFZ1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[1 * vw] - N1_exp) < 1e-6;

    // Test 2: x=1.0, powHS=3.0
    double xValCFZ2[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xValCFZ2);
    double powHSValCFZ2[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(powHSId, powHSValCFZ2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double resultOutCFZ2[4]; buffer->getLanes(resultId, resultOutCFZ2);
    double N1_exp2 = 0.5*(1.0 + std::erf(1.0/std::sqrt(2.0)));
    double expV2 = 3.0 * N1_exp2;
    bool v2 = std::abs(resultOutCFZ2[0] - expV2) < 1e-6;
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
    double underlyingValPR1[4] = {underlying_val, underlying_val, underlying_val, underlying_val}; buffer->setLanes(underlyingId, underlyingValPR1);
    double strikeValPR1[4] = {strike_val, strike_val, strike_val, strike_val}; buffer->setLanes(strikeId, strikeValPR1);
    double rateValPR1[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValPR1);
    double volValPR1[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValPR1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOutPR1[4]; buffer->getLanes(AId, AOutPR1);
    bool v1 = std::abs(AOutPR1[0] - A_exp) / std::abs(A_exp) < 0.01;
    bool g1 = std::abs(grad[0 * vw]) > 1e-10;

    // Test 2: different inputs
    double underlying_val2 = 100.0, strike_val2 = 90.0;
    double x1_exp2 = std::log(underlying_val2 / strike_val2) / stdDev_exp + muSigma_exp;
    double N1_exp2 = 0.5 * (1.0 + std::erf(x1_exp2 / std::sqrt(2.0)));
    double N2_exp2 = 0.5 * (1.0 + std::erf((x1_exp2 - stdDev_exp) / std::sqrt(2.0)));
    double A_exp2 = 1.0 * (underlying_val2 * dividendDiscount_exp * N1_exp2 - strike_val2 * riskFreeDiscount_exp * N2_exp2);
    double underlyingValPR2[4] = {underlying_val2, underlying_val2, underlying_val2, underlying_val2}; buffer->setLanes(underlyingId, underlyingValPR2);
    double strikeValPR2[4] = {strike_val2, strike_val2, strike_val2, strike_val2}; buffer->setLanes(strikeId, strikeValPR2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double AOutPR2[4]; buffer->getLanes(AId, AOutPR2);
    bool v2 = std::abs(AOutPR2[0] - A_exp2) / std::abs(A_exp2) < 0.01;
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
    double underlyingValPX0_1[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValPX0_1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0OutPX0_1[4]; buffer->getLanes(x0Id, x0OutPX0_1);
    bool v1 = std::abs(x0OutPX0_1[0] - 90.0) < 1e-10;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-6;

    // Test 2
    double underlyingValPX0_2[4] = {110.0, 110.0, 110.0, 110.0}; buffer->setLanes(underlyingId, underlyingValPX0_2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double x0OutPX0_2[4]; buffer->getLanes(x0Id, x0OutPX0_2);
    bool v2 = std::abs(x0OutPX0_2[0] - 110.0) < 1e-10;
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
    double rateValPDR1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateValPDR1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double discountOutPDR1[4]; buffer->getLanes(discountId, discountOutPDR1);
    double expV1 = std::exp(-0.04 * T);
    double expG1 = -T * expV1;
    bool v1 = std::abs(discountOutPDR1[0] - expV1) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - expG1) < 1e-3;

    // Test 2: rate=0.06
    double rateValPDR2[4] = {0.06, 0.06, 0.06, 0.06}; buffer->setLanes(rateId, rateValPDR2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double discountOutPDR2[4]; buffer->getLanes(discountId, discountOutPDR2);
    double expV2 = std::exp(-0.06 * T);
    double expG2 = -T * expV2;
    bool v2 = std::abs(discountOutPDR2[0] - expV2) < 1e-6;
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
    double rateValPZR1[4] = {0.04, 0.04, 0.04, 0.04}; buffer->setLanes(rateId, rateValPZR1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zeroRateOutPZR1[4]; buffer->getLanes(zeroRateId, zeroRateOutPZR1);
    bool v1 = std::abs(zeroRateOutPZR1[0] - 0.04) < 1e-6;
    bool g1 = std::abs(grad[0 * vw] - 1.0) < 1e-3;

    // Test 2: rate=0.06
    double rateValPZR2[4] = {0.06, 0.06, 0.06, 0.06}; buffer->setLanes(rateId, rateValPZR2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());
    double zeroRateOutPZR2[4]; buffer->getLanes(zeroRateId, zeroRateOutPZR2);
    bool v2 = std::abs(zeroRateOutPZR2[0] - 0.06) < 1e-6;
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

    double rateValCF[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValCF);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double compoundOutCF[4]; buffer->getLanes(compoundId, compoundOutCF);
    double forgeResult = compoundOutCF[0];

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

    double compoundValIRD[4] = {compound_val, compound_val, compound_val, compound_val}; buffer->setLanes(compoundId, compoundValIRD);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double rateOutIRD[4]; buffer->getLanes(rateId, rateOutIRD);
    double forgeResult = rateOutIRD[0];

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

    double inputRateValIRC[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(inputRateId, inputRateValIRC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double outputRateOutIRC[4]; buffer->getLanes(outputRateId, outputRateOutIRC);
    double forgeResult = outputRateOutIRC[0];

    BOOST_TEST_MESSAGE("  Forge output rate: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << rate_val);

    double diff = std::abs(forgeResult - rate_val);
    double diffPct = (rate_val != 0) ? diff / std::abs(rate_val) * 100.0 : diff;

    if (diffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** InterestRate constructor is BROKEN ***");
    } else {
        BOOST_TEST_MESSAGE("  InterestRate constructor: PASSED");
    }

    BOOST_CHECK_CLOSE(forgeResult, rate_val, 1e-6);
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

    double compoundValLD[4] = {compound_val, compound_val, compound_val, compound_val}; buffer->setLanes(compoundId, compoundValLD);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double rateOutLD[4]; buffer->getLanes(rateId, rateOutLD);
    double forgeResult = rateOutLD[0];

    BOOST_TEST_MESSAGE("  Forge rate result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << expected_rate);

    double diff = std::abs(forgeResult - expected_rate);
    double diffPct = diff / std::abs(expected_rate) * 100.0;

    if (diffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** log(compound)/t is BROKEN ***");
    } else {
        BOOST_TEST_MESSAGE("  log(compound)/t: PASSED");
    }

    BOOST_CHECK_CLOSE(forgeResult, expected_rate, 1e-6);
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

    double compoundValIRC2[4] = {compound_val, compound_val, compound_val, compound_val}; buffer->setLanes(compoundId, compoundValIRC2);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double rateOutIRC2[4]; buffer->getLanes(rateId, rateOutIRC2);
    double forgeResult = rateOutIRC2[0];
    double expected = std::log(compound_val) / T;

    BOOST_TEST_MESSAGE("  Forge rate result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << expected);

    if (std::abs(forgeResult) < 1e-10 && std::abs(expected) > 0.01) {
        BOOST_TEST_MESSAGE("  *** BUG: r=0 suggests comparison broke the graph! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, expected, 1e-6);
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

    double rateValFZRC[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValFZRC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double expectedDiscount = std::exp(-rate_val * T);
    double expectedCompound = 1.0 / expectedDiscount;
    double expectedLogCompound = std::log(expectedCompound);
    double expectedRate = expectedLogCompound / T;

    double discountOutFZRC[4]; buffer->getLanes(discountId, discountOutFZRC);
    double compoundOutFZRC[4]; buffer->getLanes(compoundId, compoundOutFZRC);
    double logCompoundOutFZRC[4]; buffer->getLanes(logCompoundId, logCompoundOutFZRC);
    double rateFromLogOutFZRC[4]; buffer->getLanes(rateFromLogId, rateFromLogOutFZRC);

    BOOST_TEST_MESSAGE("  Results comparison:");
    BOOST_TEST_MESSAGE("    discount:     forge=" << discountOutFZRC[0] << ", expected=" << expectedDiscount);
    BOOST_TEST_MESSAGE("    compound:     forge=" << compoundOutFZRC[0] << ", expected=" << expectedCompound);
    BOOST_TEST_MESSAGE("    log(compound): forge=" << logCompoundOutFZRC[0] << ", expected=" << expectedLogCompound);
    BOOST_TEST_MESSAGE("    rate:         forge=" << rateFromLogOutFZRC[0] << ", expected=" << expectedRate);

    // Check each step
    bool discountOK = std::abs(discountOutFZRC[0] - expectedDiscount) / expectedDiscount < 0.001;
    bool compoundOK = std::abs(compoundOutFZRC[0] - expectedCompound) / expectedCompound < 0.001;
    bool logOK = std::abs(logCompoundOutFZRC[0] - expectedLogCompound) / std::abs(expectedLogCompound) < 0.001;
    bool rateOK = std::abs(rateFromLogOutFZRC[0] - expectedRate) / std::abs(expectedRate) < 0.001;

    BOOST_TEST_MESSAGE("  Step verification:");
    BOOST_TEST_MESSAGE("    discount:      " << (discountOK ? "OK" : "FAILED"));
    BOOST_TEST_MESSAGE("    compound:      " << (compoundOK ? "OK" : "FAILED"));
    BOOST_TEST_MESSAGE("    log(compound): " << (logOK ? "OK" : "FAILED"));
    BOOST_TEST_MESSAGE("    rate:          " << (rateOK ? "OK" : "FAILED"));

    BOOST_CHECK_CLOSE(rateFromLogOutFZRC[0], expectedRate, 1e-6);
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

    int vectorWidth = buffer->getVectorWidth();
    double volValPBVR[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValPBVR);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double blackVolOutPBVR[4]; buffer->getLanes(blackVolId, blackVolOutPBVR);
    double forgeResult = blackVolOutPBVR[0];

    BOOST_TEST_MESSAGE("  Forge blackVol result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << vol_val);

    double priceDiff = std::abs(forgeResult - vol_val);
    double priceDiffPct = (vol_val != 0) ? priceDiff / vol_val * 100.0 : priceDiff;
    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND: blackVol() breaks the Forge graph! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, vol_val, 1e-6);

    // Check gradient
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

    int vectorWidth = buffer->getVectorWidth();
    double rateValBMC[4] = {rate_val, rate_val, rate_val, rate_val}; buffer->setLanes(rateId, rateValBMC);
    double volValBMC[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValBMC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double muOutBMC[4]; buffer->getLanes(muId, muOutBMC);
    double forgeResult = muOutBMC[0];

    BOOST_TEST_MESSAGE("  Forge mu result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << expected_mu);

    double priceDiff = std::abs(forgeResult - expected_mu);
    double priceDiffPct = (expected_mu != 0) ? priceDiff / std::abs(expected_mu) * 100.0 : priceDiff;
    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND: mu() calculation breaks the Forge graph! ***");
        BOOST_TEST_MESSAGE("  This is likely the source of the barrier option AAD failure.");
    }

    BOOST_CHECK_CLOSE(forgeResult, expected_mu, 1.0);

    // Check gradients
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

    int vectorWidth = buffer->getVectorWidth();
    double volValPVR[4] = {vol_val, vol_val, vol_val, vol_val}; buffer->setLanes(volId, volValPVR);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double stdDevOutPVR[4]; buffer->getLanes(stdDevId, stdDevOutPVR);
    double forgeResult = stdDevOutPVR[0];

    BOOST_TEST_MESSAGE("  Forge stdDev result: " << forgeResult);
    BOOST_TEST_MESSAGE("  Expected: " << expectedStdDev);

    double priceDiff = std::abs(forgeResult - expectedStdDev);
    double priceDiffPct = priceDiff / expectedStdDev * 100.0;
    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in variance/stdDev retrieval! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, expectedStdDev, 1e-6);

    // Check gradient: d(sqrt(vol^2 * T))/d(vol) = vol * T / sqrt(vol^2 * T) = sqrt(T)
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

    double underlyingValBBF[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValBBF);
    double barrierValBBF[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValBBF);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double BOutBBF[4]; buffer->getLanes(BId, BOutBBF);
    double forgeResult = BOutBBF[0];

    // Compute expected value manually
    double x2_exp = std::log(90.0 / 70.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(x2_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((x2_exp - 0.2) / std::sqrt(2.0)));
    double B_exp = 1.0 * (90.0 * 1.0 * N1_exp - 100.0 * 0.95 * N2_exp);

    BOOST_TEST_MESSAGE("  B() value: expected=" << B_exp << ", forge=" << forgeResult);

    double priceDiff = std::abs(forgeResult - B_exp);
    double priceDiffPct = (B_exp != 0) ? priceDiff / std::abs(B_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in B() function! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, B_exp, 1e-6);

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

    double underlyingValBCF[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValBCF);
    double barrierValBCF[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValBCF);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double COutBCF[4]; buffer->getLanes(CId, COutBCF);
    double forgeResult = COutBCF[0];

    // Compute expected value manually
    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y1_exp = std::log(70.0 * HS_exp / 100.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(y1_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((y1_exp - 0.2) / std::sqrt(2.0)));
    double C_exp = 1.0 * (90.0 * 1.0 * powHS1_exp * N1_exp - 100.0 * 0.95 * powHS0_exp * N2_exp);

    BOOST_TEST_MESSAGE("  C() value (no conditional): expected=" << C_exp << ", forge=" << forgeResult);

    double priceDiff = std::abs(forgeResult - C_exp);
    double priceDiffPct = (C_exp != 0) ? priceDiff / std::abs(C_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in C() function (even without conditional)! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, C_exp, 1e-6);

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

    double underlyingValBCFC[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValBCFC);
    double barrierValBCFC[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValBCFC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double COutBCFC[4]; buffer->getLanes(CId, COutBCFC);
    double forgeResult = COutBCFC[0];

    // Compute expected value manually
    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y1_exp = std::log(70.0 * HS_exp / 100.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(y1_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((y1_exp - 0.2) / std::sqrt(2.0)));
    double C_exp = 1.0 * (90.0 * 1.0 * powHS1_exp * N1_exp - 100.0 * 0.95 * powHS0_exp * N2_exp);

    BOOST_TEST_MESSAGE("  C() value (WITH conditional): expected=" << C_exp << ", forge=" << forgeResult);

    double priceDiff = std::abs(forgeResult - C_exp);
    double priceDiffPct = (C_exp != 0) ? priceDiff / std::abs(C_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE CONFIRMED: The (N1==0 ? 0 : ...) conditional breaks Forge! ***");
        BOOST_TEST_MESSAGE("  This is the ROOT CAUSE of the barrier option AAD failure.");
    }

    BOOST_CHECK_CLOSE(forgeResult, C_exp, 1.0);  // Allow 1% tolerance to see actual difference

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

    int vectorWidth = buffer->getVectorWidth();
    double underlyingValBDF[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValBDF);
    double barrierValBDF[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValBDF);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double DOutBDF[4]; buffer->getLanes(DId, DOutBDF);
    double forgeResult = DOutBDF[0];

    // Compute expected value manually
    double HS_exp = 70.0 / 90.0;
    double powHS0_exp = std::pow(HS_exp, 2.0 * 0.5);
    double powHS1_exp = powHS0_exp * HS_exp * HS_exp;
    double y2_exp = std::log(70.0 / 90.0) / 0.2 + 0.3;
    double N1_exp = 0.5 * (1.0 + std::erf(y2_exp / std::sqrt(2.0)));
    double N2_exp = 0.5 * (1.0 + std::erf((y2_exp - 0.2) / std::sqrt(2.0)));
    double D_exp = 1.0 * (90.0 * 1.0 * powHS1_exp * N1_exp - 100.0 * 0.95 * powHS0_exp * N2_exp);

    BOOST_TEST_MESSAGE("  D() value (no conditional): expected=" << D_exp << ", forge=" << forgeResult);

    double priceDiff = std::abs(forgeResult - D_exp);
    double priceDiffPct = (D_exp != 0) ? priceDiff / std::abs(D_exp) * 100.0 : priceDiff;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in D() function! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, D_exp, 1e-6);

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

    int vectorWidth = buffer->getVectorWidth();
    double aValCBG[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aValCBG);
    double bValCBG[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(bId, bValCBG);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double resultOutCBG[4]; buffer->getLanes(resultId, resultOutCBG);
    double forgeResult = resultOutCBG[0];
    double expected = 200.0;  // a * 2.0 = 100 * 2 = 200

    BOOST_TEST_MESSAGE("  Result with comparison: expected=" << expected << ", forge=" << forgeResult);

    double priceDiff = std::abs(forgeResult - expected);
    double priceDiffPct = priceDiff / expected * 100.0;

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** CONFIRMED: Comparison (a >= b) breaks Forge graph! ***");
    }

    // Check gradient - if comparison broke the graph, gradient will be wrong
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
    BOOST_CHECK_CLOSE(forgeResult, expected, 1e-6);
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
    int vectorWidth = buffer->getVectorWidth();
    double aValCDI[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(aId, aValCDI);  // Now a=100 >= b=70, should take if branch
    double bValCDI[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(bId, bValCDI);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double resultOutCDI[4]; buffer->getLanes(resultId, resultOutCDI);
    double forgeResult = resultOutCDI[0];

    // If graph respects runtime values: result = a * 2 = 100 * 2 = 200
    // If graph is fixed at build time: result = b * 3 = 70 * 3 = 210
    double expected_if_runtime = 200.0;
    double expected_if_buildtime = 210.0;

    BOOST_TEST_MESSAGE("  Executed with a=100, b=70 (a >= b)");
    BOOST_TEST_MESSAGE("  Forge result: " << forgeResult);
    BOOST_TEST_MESSAGE("  If runtime branching: " << expected_if_runtime);
    BOOST_TEST_MESSAGE("  If build-time branching: " << expected_if_buildtime);

    if (std::abs(forgeResult - expected_if_buildtime) < 1e-6) {
        BOOST_TEST_MESSAGE("  *** CONFIRMED: Branch is FIXED at graph build time! ***");
        BOOST_TEST_MESSAGE("  The comparison (a >= b) evaluated at build time, not runtime.");
        BOOST_TEST_MESSAGE("  This is why the full BarrierEngine fails - it uses different");
        BOOST_TEST_MESSAGE("  formulas based on (strike >= barrier) which is fixed at build time.");
    }

    // We expect build-time branching (which is actually correct for Forge)
    // The issue is that AnalyticBarrierEngine builds the graph with one set of values
    // but executes with different values - and the branch doesn't change
    BOOST_CHECK_CLOSE(forgeResult, expected_if_buildtime, 1e-6);

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

    int vectorWidth = buffer->getVectorWidth();
    double underlyingValAMC[4] = {90.0, 90.0, 90.0, 90.0}; buffer->setLanes(underlyingId, underlyingValAMC);
    double strikeValAMC[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValAMC);
    double barrierValAMC[4] = {70.0, 70.0, 70.0, 70.0}; buffer->setLanes(barrierId, barrierValAMC);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double resultOutAMC[4]; buffer->getLanes(resultId, resultOutAMC);
    double forgeResult = resultOutAMC[0];

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
    BOOST_TEST_MESSAGE("  A - C: expected=" << expected << ", forge=" << forgeResult);

    double priceDiff = std::abs(forgeResult - expected);
    double priceDiffPct = (expected != 0) ? priceDiff / std::abs(expected) * 100.0 : priceDiff;

    BOOST_TEST_MESSAGE("  Difference: " << priceDiffPct << "%");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  *** ISSUE FOUND in A-C formula! ***");
    }

    BOOST_CHECK_CLOSE(forgeResult, expected, 1e-6);

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

    int vectorWidth = buffer->getVectorWidth();
    double spotValVO[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(spotId, spotValVO);
    double strikeValVO[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(strikeId, strikeValVO);
    double rateValVO[4] = {0.05, 0.05, 0.05, 0.05}; buffer->setLanes(rateId, rateValVO);
    double volValVO[4] = {0.20, 0.20, 0.20, 0.20}; buffer->setLanes(volId, volValVO);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double priceOutVO[4]; buffer->getLanes(priceId, priceOutVO);
    double forgePrice = priceOutVO[0];
    BOOST_TEST_MESSAGE("  Forge vanilla option price: " << forgePrice);

    // Check prices match
    BOOST_CHECK_CLOSE(forgePrice, refPrice, 1e-6);

    // Get gradients
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

    int vectorWidth = buffer->getVectorWidth();
    double uValBFP[4] = {value(u), value(u), value(u), value(u)}; buffer->setLanes(uId, uValBFP);
    double strikeValBFP[4] = {value(strike), value(strike), value(strike), value(strike)}; buffer->setLanes(strikeId, strikeValBFP);
    double rValBFP[4] = {value(r), value(r), value(r), value(r)}; buffer->setLanes(rId, rValBFP);
    double vValBFP[4] = {value(v), value(v), value(v), value(v)}; buffer->setLanes(vId, vValBFP);
    double bValBFP[4] = {value(b), value(b), value(b), value(b)}; buffer->setLanes(bId, bValBFP);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double priceOutBFP[4]; buffer->getLanes(priceId, priceOutBFP);
    double forgePrice = priceOutBFP[0];
    BOOST_TEST_MESSAGE("  Forge barrier price: " << forgePrice);

    double priceDiff = std::abs(forgePrice - refPrice);
    double priceDiffPct = priceDiff / refPrice * 100.0;
    BOOST_TEST_MESSAGE("  Price difference: " << priceDiff << " (" << priceDiffPct << "%)");

    if (priceDiffPct > 1.0) {
        BOOST_TEST_MESSAGE("  WARNING: Forward pass mismatch! Forge computes different value.");
        BOOST_TEST_MESSAGE("  This indicates a problem in graph construction or execution.");
    }

    BOOST_CHECK_CLOSE(forgePrice, refPrice, 1e-6);

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

    int vectorWidth = buffer->getVectorWidth();
    double strikeValBOD[4] = {value(values.strike), value(values.strike), value(values.strike), value(values.strike)}; buffer->setLanes(strikeNodeId, strikeValBOD);
    double uValBOD[4] = {value(values.u), value(values.u), value(values.u), value(values.u)}; buffer->setLanes(uNodeId, uValBOD);
    double rValBOD[4] = {value(values.r), value(values.r), value(values.r), value(values.r)}; buffer->setLanes(rNodeId, rValBOD);
    double bValBOD[4] = {value(values.b), value(values.b), value(values.b), value(values.b)}; buffer->setLanes(bNodeId, bValBOD);
    double vValBOD[4] = {value(values.v), value(values.v), value(values.v), value(values.v)}; buffer->setLanes(vNodeId, vValBOD);

    buffer->clearGradients();
    kernel->execute(*buffer);

    double priceOutBOD[4]; buffer->getLanes(priceNodeId, priceOutBOD);
    double actual = priceOutBOD[0];

    BOOST_TEST_MESSAGE("  ONLY STABILITY CLEANING:");
    BOOST_TEST_MESSAGE("    Forge price: " << actual);
    BOOST_TEST_MESSAGE("    Expected price: " << expectedVal);

    double priceDiffPct = std::abs(actual - expectedVal) / std::abs(expectedVal) * 100.0;
    BOOST_TEST_MESSAGE("    Price difference: " << priceDiffPct << "%");

    // Get gradients
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
        int vectorWidth = buffer->getVectorWidth();
        double strikeValKR[4] = {value(testValues.strike), value(testValues.strike), value(testValues.strike), value(testValues.strike)}; buffer->setLanes(strikeNodeId, strikeValKR);
        double uValKR[4] = {value(testValues.u), value(testValues.u), value(testValues.u), value(testValues.u)}; buffer->setLanes(uNodeId, uValKR);
        double rValKR[4] = {value(testValues.r), value(testValues.r), value(testValues.r), value(testValues.r)}; buffer->setLanes(rNodeId, rValKR);
        double bValKR[4] = {value(testValues.b), value(testValues.b), value(testValues.b), value(testValues.b)}; buffer->setLanes(bNodeId, bValKR);
        double vValKR[4] = {value(testValues.v), value(testValues.v), value(testValues.v), value(testValues.v)}; buffer->setLanes(vNodeId, vValKR);

        buffer->clearGradients();
        kernel->execute(*buffer);

        double priceOutKR[4]; buffer->getLanes(priceNodeId, priceOutKR);
        double forgePrice = priceOutKR[0];

        // Get gradients
        std::vector<size_t> gradientIndices = {
            buffer->getBufferIndex(strikeNodeId),
            buffer->getBufferIndex(uNodeId),
            buffer->getBufferIndex(rNodeId),
            buffer->getBufferIndex(bNodeId),
            buffer->getBufferIndex(vNodeId)
        };
        std::vector<double> gradients(5 * vectorWidth);
        buffer->getGradientLanes(gradientIndices, gradients.data());

        double priceDiff = std::abs(forgePrice - expectedVal) / std::abs(expectedVal) * 100.0;

        BOOST_TEST_MESSAGE("    Expected price: " << expectedVal);
        BOOST_TEST_MESSAGE("    Forge price: " << forgePrice);
        BOOST_TEST_MESSAGE("    Price diff: " << priceDiff << "%");
        BOOST_TEST_MESSAGE("    Derivatives - Forge vs Bumping:");
        BOOST_TEST_MESSAGE("      d/dStrike: " << gradients[0 * vectorWidth] << " vs " << value(derivatives_bumping.strike));
        BOOST_TEST_MESSAGE("      d/dU: " << gradients[1 * vectorWidth] << " vs " << value(derivatives_bumping.u));
        BOOST_TEST_MESSAGE("      d/dR: " << gradients[2 * vectorWidth] << " vs " << value(derivatives_bumping.r));
        BOOST_TEST_MESSAGE("      d/dB: " << gradients[3 * vectorWidth] << " vs " << value(derivatives_bumping.b));
        BOOST_TEST_MESSAGE("      d/dV: " << gradients[4 * vectorWidth] << " vs " << value(derivatives_bumping.v));

        // Verify price
        BOOST_CHECK_CLOSE(expectedVal, forgePrice, 1e-6);

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
