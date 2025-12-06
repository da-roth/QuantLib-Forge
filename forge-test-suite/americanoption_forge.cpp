/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   American Option AAD test using Forge.

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
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/baroneadesiwhaleyengine.hpp>
#include <ql/pricingengines/vanilla/bjerksundstenslandengine.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesshoutengine.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/juquadraticengine.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/utilities/dataformatters.hpp>

// Forge integration headers
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <map>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibForgeRisksTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(AmericanOptionForgeTests)


namespace {

    struct AmericanOptionData {
        Option::Type type;
        Real strike;
        Real s;       // spot
        Rate q;       // dividend
        Rate r;       // risk-free rate
        Time t;       // time to maturity
        Volatility v; // volatility
    };

}

namespace {

    template <class PriceFunc>
    Real priceWithBumping(const AmericanOptionData& value,
                          AmericanOptionData& derivatives,
                          PriceFunc func) {
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.q += eps;
        auto vplus = func(data);
        derivatives.q = (vplus - v) / eps;
        data = value;

        data.r += eps;
        vplus = func(data);
        derivatives.r = (vplus - v) / eps;
        data = value;

        data.s += eps;
        vplus = func(data);
        derivatives.s = (vplus - v) / eps;
        data = value;

        data.strike += eps;
        vplus = func(data);
        derivatives.strike = (vplus - v) / eps;
        data = value;

        data.t += eps;
        vplus = func(data);
        derivatives.t = (vplus - v) / eps;
        data = value;

        data.v += eps;
        vplus = func(data);
        derivatives.v = (vplus - v) / eps;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const AmericanOptionData& values,
                           AmericanOptionData& derivatives,
                           PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.q.markForgeInputAndDiff();
        data.r.markForgeInputAndDiff();
        data.s.markForgeInputAndDiff();
        data.strike.markForgeInputAndDiff();
        data.t.markForgeInputAndDiff();
        data.v.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId qNodeId = data.q.forgeNodeId();
        forge::NodeId rNodeId = data.r.forgeNodeId();
        forge::NodeId sNodeId = data.s.forgeNodeId();
        forge::NodeId strikeNodeId = data.strike.forgeNodeId();
        forge::NodeId tNodeId = data.t.forgeNodeId();
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
        forge::ForgeEngine compiler;
        auto kernel = compiler.compile(graph);
        auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

        // Set input values
        buffer->setValue(qNodeId, value(values.q));
        buffer->setValue(rNodeId, value(values.r));
        buffer->setValue(sNodeId, value(values.s));
        buffer->setValue(strikeNodeId, value(values.strike));
        buffer->setValue(tNodeId, value(values.t));
        buffer->setValue(vNodeId, value(values.v));

        // Execute (forward + backward in one call)
        kernel->execute(*buffer);

        // Get the price value
        double priceValue = buffer->getValue(priceNodeId);

        // Get gradients directly
        int vectorWidth = buffer->getVectorWidth();
        std::vector<size_t> gradientIndices = {
            static_cast<size_t>(qNodeId) * vectorWidth,
            static_cast<size_t>(rNodeId) * vectorWidth,
            static_cast<size_t>(sNodeId) * vectorWidth,
            static_cast<size_t>(strikeNodeId) * vectorWidth,
            static_cast<size_t>(tNodeId) * vectorWidth,
            static_cast<size_t>(vNodeId) * vectorWidth
        };
        std::vector<double> gradients(6);
        buffer->getGradientsDirect(gradientIndices, gradients.data());

        derivatives.q = gradients[0];
        derivatives.r = gradients[1];
        derivatives.s = gradients[2];
        derivatives.strike = gradients[3];
        derivatives.t = gradients[4];
        derivatives.v = gradients[5];

        return Real(priceValue);
    }
}

namespace {
    Real priceBaroneAdesiWhaley(const AmericanOptionData& value) {
        Date today = Date::todaysDate();
        DayCounter dc = Actual360();
        auto spot = ext::make_shared<SimpleQuote>(0.0);
        auto qRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
        auto rRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
        auto vol = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

        auto payoff = ext::make_shared<PlainVanillaPayoff>(value.type, value.strike);
        Date exDate = today + timeToDays(value.t);
        auto exercise = ext::make_shared<AmericanExercise>(today, exDate);

        spot->setValue(value.s);
        qRate->setValue(value.q);
        rRate->setValue(value.r);
        vol->setValue(value.v);

        auto stochProcess = ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(spot), Handle<YieldTermStructure>(qTS), Handle<YieldTermStructure>(rTS),
            Handle<BlackVolTermStructure>(volTS));

        auto engine = ext::make_shared<BaroneAdesiWhaleyApproximationEngine>(stochProcess);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(engine);

        return option.NPV();
    }
}

BOOST_AUTO_TEST_CASE(testBaroneAdesiWhaleyValues) {

    BOOST_TEST_MESSAGE("Testing Barone-Adesi and Whaley approximation "
                       "for American options derivatives with Forge AAD...");

    // input
    auto data = AmericanOptionData{Option::Call, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15};

    // bump
    auto derivatives_bump = AmericanOptionData{};
    auto expected = priceWithBumping(data, derivatives_bump, priceBaroneAdesiWhaley);

    // Forge AAD
    auto derivatives_forge = AmericanOptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceBaroneAdesiWhaley);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bump.q, derivatives_forge.q, 1e-3);
    QL_CHECK_CLOSE(derivatives_bump.r, derivatives_forge.r, 1e-3);
    QL_CHECK_CLOSE(derivatives_bump.s, derivatives_forge.s, 1e-3);
    QL_CHECK_CLOSE(derivatives_bump.strike, derivatives_forge.strike, 1e-3);
    QL_CHECK_CLOSE(derivatives_bump.t, derivatives_forge.t, 1e-3);
    QL_CHECK_CLOSE(derivatives_bump.v, derivatives_forge.v, 1e-3);
}

namespace {

    Real priceBjerksundStensland(const AmericanOptionData& value) {
        Date today = Date::todaysDate();
        DayCounter dc = Actual360();
        auto spot = ext::make_shared<SimpleQuote>(0.0);
        auto qRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
        auto rRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
        auto vol = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

        auto payoff = ext::make_shared<PlainVanillaPayoff>(value.type, value.strike);
        Date exDate = today + timeToDays(value.t);
        auto exercise = ext::make_shared<AmericanExercise>(today, exDate);

        spot->setValue(value.s);
        qRate->setValue(value.q);
        rRate->setValue(value.r);
        vol->setValue(value.v);

        auto stochProcess = ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(spot), Handle<YieldTermStructure>(qTS), Handle<YieldTermStructure>(rTS),
            Handle<BlackVolTermStructure>(volTS));

        auto engine = ext::make_shared<BjerksundStenslandApproximationEngine>(stochProcess);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(engine);

        return option.NPV();
    }

}

// TODO: Re-enable when Forge supports erfc (complementary error function)
// Currently fails with: "Cannot convert active Double to passive during recording"
// because erfc_op is not handled by Forge's expression system
BOOST_AUTO_TEST_CASE(testBjerksundStenslandDerivatives, *boost::unit_test::disabled()) {
    BOOST_TEST_MESSAGE("Testing Bjerksund and Stensland approximation "
                       "for American options derivatives with Forge AAD...");

    // input
    auto data = AmericanOptionData{Option::Call, 40.00, 42.00, 0.08, 0.04, 0.75, 0.35};

    // bump
    auto derivatives_bump = AmericanOptionData{};
    auto expected = priceWithBumping(data, derivatives_bump, priceBjerksundStensland);

    // Forge AAD
    auto derivatives_forge = AmericanOptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceBjerksundStensland);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bump.q, derivatives_forge.q, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.r, derivatives_forge.r, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.s, derivatives_forge.s, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.strike, derivatives_forge.strike, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.t, derivatives_forge.t, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.v, derivatives_forge.v, 1e-4);
}

namespace {

    Real priceJu(const AmericanOptionData& juValue) {
        Date today = Date::todaysDate();
        DayCounter dc = Actual360();
        auto spot = ext::make_shared<SimpleQuote>(0.0);
        auto qRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
        auto rRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
        auto vol = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

        auto payoff = ext::make_shared<PlainVanillaPayoff>(juValue.type, juValue.strike);
        Date exDate = today + timeToDays(juValue.t);
        ext::shared_ptr<Exercise> exercise = ext::make_shared<AmericanExercise>(today, exDate);

        spot->setValue(juValue.s);
        qRate->setValue(juValue.q);
        rRate->setValue(juValue.r);
        vol->setValue(juValue.v);

        auto stochProcess = ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(spot), Handle<YieldTermStructure>(qTS), Handle<YieldTermStructure>(rTS),
            Handle<BlackVolTermStructure>(volTS));

        auto engine = ext::make_shared<JuQuadraticApproximationEngine>(stochProcess);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(engine);

        return option.NPV();
    }
}

BOOST_AUTO_TEST_CASE(testJuDerivatives) {
    BOOST_TEST_MESSAGE("Testing Ju approximation for American options derivatives with Forge AAD...");

    // input
    auto data = AmericanOptionData{Option::Call, 100.00, 80.00, 0.07, 0.03, 3.0, 0.2};

    // bump
    auto derivatives_bump = AmericanOptionData{};
    auto expected = priceWithBumping(data, derivatives_bump, priceJu);

    // Forge AAD
    auto derivatives_forge = AmericanOptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceJu);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bump.q, derivatives_forge.q, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.r, derivatives_forge.r, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.s, derivatives_forge.s, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.strike, derivatives_forge.strike, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.t, derivatives_forge.t, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.v, derivatives_forge.v, 1e-4);
}

namespace {
    Real priceFd(const AmericanOptionData& juValue) {
        Date today = Date::todaysDate();
        DayCounter dc = Actual360();
        auto spot = ext::make_shared<SimpleQuote>(0.0);
        auto qRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
        auto rRate = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
        auto vol = ext::make_shared<SimpleQuote>(0.0);
        ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

        auto payoff = ext::make_shared<PlainVanillaPayoff>(juValue.type, juValue.strike);

        Date exDate = today + timeToDays(juValue.t);
        auto exercise = ext::make_shared<AmericanExercise>(today, exDate);

        spot->setValue(juValue.s);
        qRate->setValue(juValue.q);
        rRate->setValue(juValue.r);
        vol->setValue(juValue.v);

        ext::shared_ptr<BlackScholesMertonProcess> stochProcess =
            ext::make_shared<BlackScholesMertonProcess>(
                Handle<Quote>(spot), Handle<YieldTermStructure>(qTS),
                Handle<YieldTermStructure>(rTS), Handle<BlackVolTermStructure>(volTS));

        ext::shared_ptr<PricingEngine> engine =
            ext::make_shared<FdBlackScholesVanillaEngine>(stochProcess, 100, 100);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(engine);

        return option.NPV();
    }

}

// TODO: Re-enable when Forge supports asinh, sinh, scalar_min, scalar_max operations
// Currently fails with: "Cannot convert active Double to passive during recording"
// because these operations fall back and break the Forge graph
BOOST_AUTO_TEST_CASE(testFdDerivatives, *boost::unit_test::disabled()) {
    BOOST_TEST_MESSAGE("Testing finite-difference engine "
                       "for American options derivatives with Forge AAD...");

    // input
    auto data = AmericanOptionData{Option::Call, 100.00, 80.00, 0.07, 0.03, 3.0, 0.2};

    // bump
    auto derivatives_bump = AmericanOptionData{};
    auto expected = priceWithBumping(data, derivatives_bump, priceFd);

    // Forge AAD
    auto derivatives_forge = AmericanOptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceFd);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bump.q, derivatives_forge.q, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.r, derivatives_forge.r, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.s, derivatives_forge.s, 1e-3);
    QL_CHECK_CLOSE(derivatives_bump.strike, derivatives_forge.strike, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.t, derivatives_forge.t, 1e-4);
    QL_CHECK_CLOSE(derivatives_bump.v, derivatives_forge.v, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
