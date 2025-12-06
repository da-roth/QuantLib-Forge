/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Bates Model AAD test using Forge.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors
   Copyright (C) 2005, 2008 Klaus Spanderen
   Copyright (C) 2007 StatPro Italia srl

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2022 Xcelerit

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/


#include "toplevelfixture.hpp"
#include "utilities_forge.hpp"
#include <ql/instruments/europeanoption.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/models/equity/batesmodel.hpp>
#include <ql/models/equity/hestonmodelhelper.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/batesengine.hpp>
#include <ql/pricingengines/vanilla/fdbatesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/jumpdiffusionengine.hpp>
#include <ql/pricingengines/vanilla/mceuropeanhestonengine.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/merton76process.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/period.hpp>

// Forge integration headers
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibForgeRisksTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(BatesModelForgeTests)

namespace {

    // This is a copy of the bates pricing from the main test suite,
    // separated as a function of independent variables to allow bump vs aad tests
    Real priceBatesModel(Real riskFreeRate, Real dividendRate, Real strike) {
        Date settlementDate = Date::todaysDate();
        Settings::instance().evaluationDate() = settlementDate;

        DayCounter dayCounter = ActualActual(ActualActual::ISDA);
        Date exerciseDate = settlementDate + 6 * Months;

        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 30);
        auto exercise = ext::make_shared<EuropeanExercise>(exerciseDate);

        Handle<YieldTermStructure> riskFreeTS(flatRate(riskFreeRate, dayCounter));
        Handle<YieldTermStructure> dividendTS(flatRate(dividendRate, dayCounter));
        Handle<Quote> s0(ext::make_shared<SimpleQuote>(strike));

        Real yearFraction = dayCounter.yearFraction(settlementDate, exerciseDate);
        Real forwardPrice = s0->value() * std::exp((0.1 - 0.04) * yearFraction);

        const Real v0 = 0.05;
        const Real kappa = 5.0;
        const Real theta = 0.05;
        const Real sigma = 1.0e-4;
        const Real rho = 0.0;
        const Real lambda = 0.0001;
        const Real nu = 0.0;
        const Real delta = 0.0001;

        VanillaOption option(payoff, exercise);

        auto process = ext::make_shared<BatesProcess>(riskFreeTS, dividendTS, s0, v0, kappa, theta,
                                                      sigma, rho, lambda, nu, delta);

        auto engine = ext::make_shared<BatesEngine>(ext::make_shared<BatesModel>(process), 64);

        option.setPricingEngine(engine);
        return option.NPV();
    }

    Real priceBatesModelBumping(Real riskFreeRate,
                                Real dividendRate,
                                Real strike,
                                std::vector<Real>& der) {
        Real eps = 1e-7;
        auto v = priceBatesModel(riskFreeRate, dividendRate, strike);
        auto vplus = priceBatesModel(riskFreeRate + eps, dividendRate, strike);
        der.push_back((vplus - v) / eps);
        vplus = priceBatesModel(riskFreeRate, dividendRate + eps, strike);
        der.push_back((vplus - v) / eps);
        vplus = priceBatesModel(riskFreeRate, dividendRate, strike + eps);
        der.push_back((vplus - v) / eps);
        return v;
    }
}

// TODO: Re-enable when Forge supports sin, cos, atan2, hypot, scalar_max, scalar_min operations
// Currently fails with: "Cannot convert active Double to passive during recording"
// because the Bates/Heston complex number arithmetic requires trig functions not yet in Forge
BOOST_AUTO_TEST_CASE(testBatesModelDerivatives, *boost::unit_test::disabled()) {
    BOOST_TEST_MESSAGE("Testing Bates Model derivatives with Forge AAD...");

    SavedSettings backup;

    double riskFreeRateVal = 0.1;
    double dividendRateVal = 0.04;
    double strikeVal = 32.0;

    // bumping
    std::vector<Real> gradient_bump;
    auto expected = priceBatesModelBumping(riskFreeRateVal, dividendRateVal, strikeVal, gradient_bump);

    // Forge AAD
    forge::GraphRecorder recorder;
    recorder.start();

    Real riskFreeRate = riskFreeRateVal;
    Real dividendRate = dividendRateVal;
    Real strike = strikeVal;

    riskFreeRate.markForgeInputAndDiff();
    dividendRate.markForgeInputAndDiff();
    strike.markForgeInputAndDiff();

    forge::NodeId riskFreeRateNodeId = riskFreeRate.forgeNodeId();
    forge::NodeId dividendRateNodeId = dividendRate.forgeNodeId();
    forge::NodeId strikeNodeId = strike.forgeNodeId();

    Real price = priceBatesModel(riskFreeRate, dividendRate, strike);

    price.markForgeOutput();
    forge::NodeId priceNodeId = price.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vectorWidth = buffer->getVectorWidth();
    double riskFreeRateIn[4] = {riskFreeRateVal, riskFreeRateVal, riskFreeRateVal, riskFreeRateVal}; buffer->setLanes(riskFreeRateNodeId, riskFreeRateIn);
    double dividendRateIn[4] = {dividendRateVal, dividendRateVal, dividendRateVal, dividendRateVal}; buffer->setLanes(dividendRateNodeId, dividendRateIn);
    double strikeIn[4] = {strikeVal, strikeVal, strikeVal, strikeVal}; buffer->setLanes(strikeNodeId, strikeIn);

    kernel->execute(*buffer);

    double priceOut[4]; buffer->getLanes(priceNodeId, priceOut);
    double priceValue = priceOut[0];

    std::vector<size_t> gradientIndices = {
        buffer->getBufferIndex(riskFreeRateNodeId),
        buffer->getBufferIndex(dividendRateNodeId),
        buffer->getBufferIndex(strikeNodeId)
    };
    std::vector<double> gradients(3 * vectorWidth);
    buffer->getGradientLanes(gradientIndices, gradients.data());

    // compare
    QL_CHECK_CLOSE(expected, Real(priceValue), 1e-9);
    QL_CHECK_CLOSE(gradient_bump[0], Real(gradients[0 * vectorWidth]), 1e-4);
    QL_CHECK_CLOSE(gradient_bump[1], Real(gradients[1 * vectorWidth]), 1e-4);
    QL_CHECK_CLOSE(gradient_bump[2], Real(gradients[2 * vectorWidth]), 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
