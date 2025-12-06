/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Swap AAD test using Forge.

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
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/time/daycounters/simpledaycounter.hpp>

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

BOOST_AUTO_TEST_SUITE(SwapForgeTests)

namespace {

    struct SwapData {
        VanillaSwap::Type type;
        Real n;       // nominal
        Real s;       // spread
        Rate g;       // gearing
        Volatility v; // volatility
    };

}

namespace {

    template <class PriceFunc>
    Real priceWithBumping(const SwapData& value, SwapData& derivatives, PriceFunc func) {
        // bumping
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.n += 1;
        auto vplus = func(data);
        derivatives.n = (vplus - v) / 1;
        data = value;

        data.s += eps;
        vplus = func(data);
        derivatives.s = (vplus - v) / eps;
        data = value;

        data.g += eps;
        vplus = func(data);
        derivatives.g = (vplus - v) / eps;
        data = value;


        data.v += eps;
        vplus = func(data);
        derivatives.v = (vplus - v) / eps;
        data = value;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const SwapData& values, SwapData& derivatives, PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.n.markForgeInputAndDiff();
        data.s.markForgeInputAndDiff();
        data.g.markForgeInputAndDiff();
        data.v.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId nNodeId = data.n.forgeNodeId();
        forge::NodeId sNodeId = data.s.forgeNodeId();
        forge::NodeId gNodeId = data.g.forgeNodeId();
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
        buffer->setValue(nNodeId, value(values.n));
        buffer->setValue(sNodeId, value(values.s));
        buffer->setValue(gNodeId, value(values.g));
        buffer->setValue(vNodeId, value(values.v));

        // Execute (forward + backward in one call)
        buffer->clearGradients();
        kernel->execute(*buffer);

        // Get the price value
        double priceOutput[1];
        buffer->getLanes(priceNodeId, priceOutput);
        double priceValue = priceOutput[0];

        // Get gradients directly
        int vectorWidth = buffer->getVectorWidth();
        std::vector<size_t> gradientIndices = {
            static_cast<size_t>(nNodeId) * vectorWidth,
            static_cast<size_t>(sNodeId) * vectorWidth,
            static_cast<size_t>(gNodeId) * vectorWidth,
            static_cast<size_t>(vNodeId) * vectorWidth
        };
        std::vector<double> gradients(4);
        double* gradOutputs[4] = {gradients.data(), nullptr, nullptr, nullptr};
        buffer->getGradientLanes(gradientIndices, gradOutputs);

        derivatives.n = gradients[0];
        derivatives.s = gradients[1];
        derivatives.g = gradients[2];
        derivatives.v = gradients[3];

        return Real(priceValue);
    }
}

namespace {
    Real priceSwap(const SwapData& value) {
        RelinkableHandle<YieldTermStructure> termStructure;
        Calendar calendar = NullCalendar();
        Date today = calendar.adjust(Settings::instance().evaluationDate());
        Date settlement = calendar.advance(today, 2, Days);
        termStructure.linkTo(flatRate(settlement, 0.05, Actual365Fixed()));
        Date maturity = today + 5 * Years;
        Natural fixingDays = 0;

        Schedule schedule(today, maturity, Period(Annual), calendar, Following, Following,
                          DateGeneration::Forward, false);
        DayCounter dayCounter = SimpleDayCounter();
        auto index = ext::make_shared<IborIndex>("dummy", 1 * Years, 0, EURCurrency(), calendar,
                                                 Following, false, dayCounter, termStructure);
        Rate oneYear = 0.05;
        Rate r = std::log(1.0 + oneYear);
        termStructure.linkTo(flatRate(today, r, dayCounter));


        std::vector<Rate> coupons(1, oneYear);
        Leg fixedLeg =
            FixedRateLeg(schedule).withNotionals(value.n).withCouponRates(coupons, dayCounter);

        Handle<OptionletVolatilityStructure> vol(ext::make_shared<ConstantOptionletVolatility>(
            today, NullCalendar(), Following, value.v, dayCounter));
        auto pricer = ext::make_shared<BlackIborCouponPricer>(vol);

        Leg floatingLeg = IborLeg(schedule, index)
                              .withNotionals(value.n)
                              .withPaymentDayCounter(dayCounter)
                              .withFixingDays(fixingDays)
                              .withGearings(value.g)
                              .withSpreads(value.s)
                              .inArrears();
        setCouponPricer(floatingLeg, pricer);

        auto swap = ext::make_shared<Swap>(floatingLeg, fixedLeg);
        swap->setPricingEngine(
            ext::shared_ptr<PricingEngine>(new DiscountingSwapEngine(termStructure)));

        return swap->NPV();
    }
}
BOOST_AUTO_TEST_CASE(testSwapDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing swap derivatives with Forge AAD...");

    // input
    auto data = SwapData{VanillaSwap::Payer, 1000000.0, -0.001, 0.01, 0.22};

    // Bumping
    auto derivatives_Bumping = SwapData{};
    auto expected = priceWithBumping(data, derivatives_Bumping, priceSwap);

    // Forge AAD
    auto derivatives_forge = SwapData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceSwap);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_Bumping.n, derivatives_forge.n, 1e-3);
    QL_CHECK_CLOSE(derivatives_Bumping.s, derivatives_forge.s, 1e-3);
    QL_CHECK_CLOSE(derivatives_Bumping.g, derivatives_forge.g, 1e-3);
    QL_CHECK_CLOSE(derivatives_Bumping.v, derivatives_forge.v, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
