/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Bermudan Swaption AAD test using Forge.

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
#include <ql/cashflows/coupon.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swaption/treeswaptionengine.hpp>
#include <ql/time/dategenerationrule.hpp>
#include <ql/time/daycounters/thirty360.hpp>
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

BOOST_AUTO_TEST_SUITE(BermudanSwaptionForgeTests)

namespace {

    struct BermudanSwaptionData {
        Swap::Type type;
        Real nominal = 1000.0;
        Real fixedRate;
        Real forwardRate;
        Real a = 0.048696;
        Real sigma = 0.0058904;
    };

}
namespace {

    template <class PriceFunc>
    Real priceWithBumping(const BermudanSwaptionData& value,
                          BermudanSwaptionData& derivatives,
                          PriceFunc func) {
        // Bumping
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.nominal += eps;
        auto vplus = func(data);
        derivatives.nominal = (vplus - v) / eps;
        data = value;

        data.fixedRate += eps;
        vplus = func(data);
        derivatives.fixedRate = (vplus - v) / eps;
        data = value;

        data.forwardRate += eps;
        vplus = func(data);
        derivatives.forwardRate = (vplus - v) / eps;
        data = value;

        data.a += eps;
        vplus = func(data);
        derivatives.a = (vplus - v) / eps;
        data = value;

        data.sigma += eps * .1;
        vplus = func(data);
        derivatives.sigma = (vplus - v) / eps / .1;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const BermudanSwaptionData& values,
                           BermudanSwaptionData& derivatives,
                           PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.nominal.markForgeInputAndDiff();
        data.fixedRate.markForgeInputAndDiff();
        data.forwardRate.markForgeInputAndDiff();
        data.a.markForgeInputAndDiff();
        data.sigma.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId nominalNodeId = data.nominal.forgeNodeId();
        forge::NodeId fixedRateNodeId = data.fixedRate.forgeNodeId();
        forge::NodeId forwardRateNodeId = data.forwardRate.forgeNodeId();
        forge::NodeId aNodeId = data.a.forgeNodeId();
        forge::NodeId sigmaNodeId = data.sigma.forgeNodeId();

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
        buffer->setValue(nominalNodeId, value(values.nominal));
        buffer->setValue(fixedRateNodeId, value(values.fixedRate));
        buffer->setValue(forwardRateNodeId, value(values.forwardRate));
        buffer->setValue(aNodeId, value(values.a));
        buffer->setValue(sigmaNodeId, value(values.sigma));

        // Execute (forward + backward in one call)
        kernel->execute(*buffer);

        // Get the price value
        double priceOutput[1];
        buffer->getLanes(priceNodeId, priceOutput);
        double priceValue = priceOutput[0];

        // Get gradients directly
        int vectorWidth = buffer->getVectorWidth();
        std::vector<size_t> gradientIndices = {
            static_cast<size_t>(nominalNodeId) * vectorWidth,
            static_cast<size_t>(fixedRateNodeId) * vectorWidth,
            static_cast<size_t>(forwardRateNodeId) * vectorWidth,
            static_cast<size_t>(aNodeId) * vectorWidth,
            static_cast<size_t>(sigmaNodeId) * vectorWidth
        };
        std::vector<double> gradients(5);
        double* gradOutputs[4] = {gradients.data(), nullptr, nullptr, nullptr};
        buffer->getGradientLanes(gradientIndices, gradOutputs);

        derivatives.nominal = gradients[0];
        derivatives.fixedRate = gradients[1];
        derivatives.forwardRate = gradients[2];
        derivatives.a = gradients[3];
        derivatives.sigma = gradients[4];

        return Real(priceValue);
    }
}

namespace {
    Real priceBermudanSwaption(const BermudanSwaptionData& value) {
        Integer startYears = 1;
        Integer length = 5;
        Natural settlementDays = 2;
        RelinkableHandle<YieldTermStructure> termStructure;
        termStructure.linkTo(
            flatRate(Date(15, February, 2002), value.forwardRate, Actual365Fixed()));
        auto index = ext::make_shared<Euribor6M>(termStructure);
        Calendar calendar = index->fixingCalendar();
        Date today = calendar.adjust(Date::todaysDate());
        Date settlement = calendar.advance(today, settlementDays, Days);

        Date start = calendar.advance(settlement, startYears, Years);
        Date maturity = calendar.advance(start, length, Years);
        Schedule fixedSchedule(start, maturity, Period(Annual), calendar, Unadjusted, Unadjusted,
                               DateGeneration::Forward, false);
        Schedule floatSchedule(start, maturity, Period(Semiannual), calendar, ModifiedFollowing,
                               ModifiedFollowing, DateGeneration::Forward, false);
        ext::shared_ptr<VanillaSwap> swap(new VanillaSwap(
            value.type, value.nominal, fixedSchedule, value.fixedRate,
            Thirty360(Thirty360::BondBasis), floatSchedule, index, 0.0, index->dayCounter()));
        swap->setPricingEngine(
            ext::shared_ptr<PricingEngine>(new DiscountingSwapEngine(termStructure)));
        auto model = ext::make_shared<HullWhite>(termStructure, value.a, value.sigma);

        std::vector<Date> exerciseDates;
        const Leg& leg = swap->fixedLeg();
        for (const auto& i : leg) {
            auto coupon = ext::dynamic_pointer_cast<Coupon>(i);
            exerciseDates.push_back(coupon->accrualStartDate());
        }
        auto exercise = ext::make_shared<BermudanExercise>(exerciseDates);

        auto treeEngine = ext::make_shared<TreeSwaptionEngine>(model, 50);

        Swaption swaption(swap, exercise);
        swaption.setPricingEngine(treeEngine);

        return swaption.NPV();
    }
}

BOOST_AUTO_TEST_CASE(testBermudanSwaptionDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing bermudan swaption derivatives with Forge AAD...");

    // input
    auto data = BermudanSwaptionData{Swap::Payer, 1000.00, 0.10, 0.04875825, 0.048696, 0.0058904};

    // bumping
    auto derivatives_bumping = BermudanSwaptionData{};
    auto expected = priceWithBumping(data, derivatives_bumping, priceBermudanSwaption);

    // Forge AAD
    auto derivatives_forge = BermudanSwaptionData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceBermudanSwaption);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    if (derivatives_bumping.nominal > 0.1)
        QL_CHECK_CLOSE(derivatives_bumping.nominal, derivatives_forge.nominal, 1e-2);
    else
        QL_CHECK_SMALL(abs(derivatives_forge.nominal - derivatives_bumping.nominal), 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.fixedRate, derivatives_forge.fixedRate, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.forwardRate, derivatives_forge.forwardRate, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.a, derivatives_forge.a, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.sigma, derivatives_forge.sigma, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
