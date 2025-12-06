/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   European Option AAD test using Forge.

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
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>

// Forge integration headers
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibForgeRisksTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(EuropeanOptionForgeTests)

namespace {

    struct EuropeanOptionData {
        Option::Type type;
        Real strike;
        Real u;       // underlying
        Rate r;       // risk-free rate
        Real d;       // dividend Yield
        Volatility v; // volatility
    };

}
namespace {

    template <class PriceFunc>
    Real priceWithAnalytics(const EuropeanOptionData& value,
                            EuropeanOptionData& derivatives,
                            PriceFunc func) {
        auto data = value;
        auto v = func(data);

        derivatives.u = v[4];
        derivatives.strike = v[2];
        derivatives.r = v[1];
        derivatives.v = v[3];
        derivatives.d = v[5];

        return v[0];
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const EuropeanOptionData& values,
                           EuropeanOptionData& derivatives,
                           PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.d.markForgeInputAndDiff();
        data.r.markForgeInputAndDiff();
        data.strike.markForgeInputAndDiff();
        data.u.markForgeInputAndDiff();
        data.v.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId dNodeId = data.d.forgeNodeId();
        forge::NodeId rNodeId = data.r.forgeNodeId();
        forge::NodeId strikeNodeId = data.strike.forgeNodeId();
        forge::NodeId uNodeId = data.u.forgeNodeId();
        forge::NodeId vNodeId = data.v.forgeNodeId();

        // Compute the price (this builds the computation graph)
        auto price = func(data)[0];

        // Mark output
        price.markForgeOutput();
        forge::NodeId priceNodeId = price.forgeNodeId();

        // Stop recording and get the graph
        recorder.stop();
        forge::Graph graph = recorder.graph();

        // JIT compile the graph
        forge::ForgeEngine compiler;
        auto kernel = compiler.compile(graph);
        auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

        // Set input values
        int vectorWidth = buffer->getVectorWidth();
        double dVal[4] = {value(values.d), value(values.d), value(values.d), value(values.d)}; buffer->setLanes(dNodeId, dVal);
        double rVal[4] = {value(values.r), value(values.r), value(values.r), value(values.r)}; buffer->setLanes(rNodeId, rVal);
        double strikeVal[4] = {value(values.strike), value(values.strike), value(values.strike), value(values.strike)}; buffer->setLanes(strikeNodeId, strikeVal);
        double uVal[4] = {value(values.u), value(values.u), value(values.u), value(values.u)}; buffer->setLanes(uNodeId, uVal);
        double vVal[4] = {value(values.v), value(values.v), value(values.v), value(values.v)}; buffer->setLanes(vNodeId, vVal);

        // Execute (forward + backward in one call)
        kernel->execute(*buffer);

        // Get the price value
        double priceOut[4]; buffer->getLanes(priceNodeId, priceOut);
        double priceValue = priceOut[0];

        // Get gradients directly
        std::vector<size_t> gradientIndices = {
            buffer->getBufferIndex(dNodeId),
            buffer->getBufferIndex(rNodeId),
            buffer->getBufferIndex(strikeNodeId),
            buffer->getBufferIndex(uNodeId),
            buffer->getBufferIndex(vNodeId)
        };
        std::vector<double> gradients(5 * vectorWidth);
        buffer->getGradientLanes(gradientIndices, gradients.data());

        derivatives.d = gradients[0 * vectorWidth];
        derivatives.r = gradients[1 * vectorWidth];
        derivatives.strike = gradients[2 * vectorWidth];
        derivatives.u = gradients[3 * vectorWidth];
        derivatives.v = gradients[4 * vectorWidth];

        return Real(priceValue);
    }
}

namespace {
    std::array<Real, 6> priceEuropeanOption(const EuropeanOptionData& value) {
        // set up dates
        Calendar calendar = TARGET();
        Date todaysDate(15, May, 1998);
        Date settlementDate(17, May, 1998);
        Settings::instance().evaluationDate() = todaysDate;

        // our options
        Option::Type type(Option::Put);
        DayCounter dayCounter = Actual365Fixed();
        Date maturity(17, May, 1999);
        auto exercise = ext::make_shared<EuropeanExercise>(maturity);

        Handle<Quote> underlyingH(ext::make_shared<SimpleQuote>(value.u));

        // bootstrap the yield/dividend/vol curves
        Handle<YieldTermStructure> flatTermStructure(
            ext::make_shared<FlatForward>(settlementDate, value.r, dayCounter));
        Handle<YieldTermStructure> flatDividendTS(
            ext::make_shared<FlatForward>(settlementDate, value.d, dayCounter));
        Handle<BlackVolTermStructure> flatVolTS(
            ext::make_shared<BlackConstantVol>(settlementDate, calendar, value.v, dayCounter));
        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, value.strike);
        auto bsmProcess = ext::make_shared<BlackScholesMertonProcess>(underlyingH, flatDividendTS,
                                                                      flatTermStructure, flatVolTS);

        // option
        auto european = ext::make_shared<VanillaOption>(payoff, exercise);
        // computing the option price with the analytic Black-Scholes formulae
        european->setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(bsmProcess));
        std::array<Real, 6> f_array = {
            european->NPV(),  european->rho(),   european->strikeSensitivity(),
            european->vega(), european->delta(), european->dividendRho()};
        return f_array;
    }
}

BOOST_AUTO_TEST_CASE(testEuropeanOptionDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing European options derivatives with Forge AAD...");

    // input
    auto data = EuropeanOptionData{Option::Call, 100.00, 90.00, 0.10, 0.10, 0.10};

    // analytics
    auto derivatives_analytics = EuropeanOptionData{};
    auto expected = priceWithAnalytics(data, derivatives_analytics, priceEuropeanOption);

    // Forge AAD
    auto derivatives_forge = EuropeanOptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceEuropeanOption);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_analytics.d, derivatives_forge.d, 1e-7);
    QL_CHECK_CLOSE(derivatives_analytics.r, derivatives_forge.r, 1e-7);
    QL_CHECK_CLOSE(derivatives_analytics.u, derivatives_forge.u, 1e-7);
    QL_CHECK_CLOSE(derivatives_analytics.strike, derivatives_forge.strike, 1e-7);
    QL_CHECK_CLOSE(derivatives_analytics.v, derivatives_forge.v, 1e-7);
}


BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
