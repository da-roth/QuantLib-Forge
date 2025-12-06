/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Credit Default Swap AAD test using Forge.

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
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/defaulttermstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/dategenerationrule.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/schedule.hpp>

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

BOOST_AUTO_TEST_SUITE(CreditDefaultSwapForgeTests)


namespace {

    struct CreditDefaultSwapData {
        // Build the CDS
        Rate fixedRate;
        Real notional;
        Real recoveryRate;
        Real hazardRate;
        Real riskFreeRate;
    };

}

namespace {

    template <class PriceFunc>
    Real priceWithBumping(const CreditDefaultSwapData& value,
                          CreditDefaultSwapData& derivatives,
                          PriceFunc func) {
        // Bumping
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.fixedRate += eps;
        auto vplus = func(data);
        derivatives.fixedRate = (vplus - v) / eps;
        data = value;

        data.notional += 1;
        vplus = func(data);
        derivatives.notional = (vplus - v) / 1;
        data = value;

        data.recoveryRate += eps;
        vplus = func(data);
        derivatives.recoveryRate = (vplus - v) / eps;
        data = value;

        data.hazardRate += eps;
        vplus = func(data);
        derivatives.hazardRate = (vplus - v) / eps;
        data = value;

        data.riskFreeRate += eps;
        vplus = func(data);
        derivatives.riskFreeRate = (vplus - v) / eps;
        data = value;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const CreditDefaultSwapData& values,
                           CreditDefaultSwapData& derivatives,
                           PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.notional.markForgeInputAndDiff();
        data.hazardRate.markForgeInputAndDiff();
        data.recoveryRate.markForgeInputAndDiff();
        data.fixedRate.markForgeInputAndDiff();
        data.riskFreeRate.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId notionalNodeId = data.notional.forgeNodeId();
        forge::NodeId hazardRateNodeId = data.hazardRate.forgeNodeId();
        forge::NodeId recoveryRateNodeId = data.recoveryRate.forgeNodeId();
        forge::NodeId fixedRateNodeId = data.fixedRate.forgeNodeId();
        forge::NodeId riskFreeRateNodeId = data.riskFreeRate.forgeNodeId();

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
        buffer->setValue(notionalNodeId, value(values.notional));
        buffer->setValue(hazardRateNodeId, value(values.hazardRate));
        buffer->setValue(recoveryRateNodeId, value(values.recoveryRate));
        buffer->setValue(fixedRateNodeId, value(values.fixedRate));
        buffer->setValue(riskFreeRateNodeId, value(values.riskFreeRate));

        // Execute (forward + backward in one call)
        kernel->execute(*buffer);

        // Get the price value
        double priceValue = buffer->getValue(priceNodeId);

        // Get gradients directly
        int vectorWidth = buffer->getVectorWidth();
        std::vector<size_t> gradientIndices = {
            static_cast<size_t>(notionalNodeId) * vectorWidth,
            static_cast<size_t>(hazardRateNodeId) * vectorWidth,
            static_cast<size_t>(recoveryRateNodeId) * vectorWidth,
            static_cast<size_t>(fixedRateNodeId) * vectorWidth,
            static_cast<size_t>(riskFreeRateNodeId) * vectorWidth
        };
        std::vector<double> gradients(5);
        buffer->getGradientsDirect(gradientIndices, gradients.data());

        derivatives.notional = gradients[0];
        derivatives.hazardRate = gradients[1];
        derivatives.recoveryRate = gradients[2];
        derivatives.fixedRate = gradients[3];
        derivatives.riskFreeRate = gradients[4];

        return Real(priceValue);
    }
}

namespace {
    Real priceCreditDefaultSwap(const CreditDefaultSwapData& value) {
        DayCounter dayCount = Actual360();
        // Initialize curves
        Settings::instance().evaluationDate() = Date(9, June, 2006);
        Date today = Settings::instance().evaluationDate();
        Calendar calendar = TARGET();

        Handle<Quote> hazardRate(ext::make_shared<SimpleQuote>(value.hazardRate));
        RelinkableHandle<DefaultProbabilityTermStructure> probabilityCurve;
        probabilityCurve.linkTo(
            ext::make_shared<FlatHazardRate>(0, calendar, hazardRate, Actual360()));

        RelinkableHandle<YieldTermStructure> discountCurve;

        discountCurve.linkTo(ext::make_shared<FlatForward>(today, value.riskFreeRate, Actual360()));

        // Build the schedule
        Date issueDate = calendar.advance(today, -1, Years);
        Date maturity = calendar.advance(issueDate, 10, Years);
        Frequency frequency = Semiannual;
        BusinessDayConvention convention = ModifiedFollowing;

        Schedule schedule(issueDate, maturity, Period(frequency), calendar, convention, convention,
                          DateGeneration::Forward, false);

        auto cds =
            ext::make_shared<CreditDefaultSwap>(Protection::Seller, value.notional, value.fixedRate,
                                                schedule, convention, dayCount, true, true);
        cds->setPricingEngine(ext::make_shared<MidPointCdsEngine>(
            probabilityCurve, value.recoveryRate, discountCurve));
        return cds->NPV();
    }
}

BOOST_AUTO_TEST_CASE(testCreditDefaultSwapDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing credit default swap derivatives with Forge AAD...");

    // input
    auto data = CreditDefaultSwapData{0.0120, 10000.0, 0.4, 0.01234, 0.06};

    // bumping
    auto derivatives_bumping = CreditDefaultSwapData{};
    auto expected = priceWithBumping(data, derivatives_bumping, priceCreditDefaultSwap);

    // Forge AAD
    auto derivatives_forge = CreditDefaultSwapData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceCreditDefaultSwap);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bumping.notional, derivatives_forge.notional, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.hazardRate, derivatives_forge.hazardRate, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.fixedRate, derivatives_forge.fixedRate, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.recoveryRate, derivatives_forge.recoveryRate, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.riskFreeRate, derivatives_forge.riskFreeRate, 1e-3);
}


BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
