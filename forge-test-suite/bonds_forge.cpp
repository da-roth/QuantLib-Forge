/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Bonds AAD test using Forge.

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
#include <ql/experimental/callablebonds/callablebond.hpp>
#include <ql/experimental/callablebonds/treecallablebondengine.hpp>
#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/dategenerationrule.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

// Forge integration headers
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibForgeRisksTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(BondsForgeTests)

namespace {

    struct BondsData {
        Real spotRate1;
        Real spotRate2;
        Real spotRate3;
        Real couponRate;
        Real faceValue;
    };

}
namespace {

    template <class PriceFunc>
    Real priceWithBumping(const BondsData& value, BondsData& derivatives, PriceFunc func) {
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.spotRate1 += eps;
        auto vplus = func(data);
        derivatives.spotRate1 = (vplus - v) / eps;
        data = value;

        data.spotRate2 += eps;
        vplus = func(data);
        derivatives.spotRate2 = (vplus - v) / eps;
        data = value;

        data.spotRate3 += eps;
        vplus = func(data);
        derivatives.spotRate3 = (vplus - v) / eps;
        data = value;

        data.couponRate += eps;
        vplus = func(data);
        derivatives.couponRate = (vplus - v) / eps;
        data = value;

        data.faceValue += eps;
        vplus = func(data);
        derivatives.faceValue = (vplus - v) / eps;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const BondsData& values, BondsData& derivatives, PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create FRESH Real inputs, assign values, then mark for differentiation
        Real spotRate1 = value(values.spotRate1);
        Real spotRate2 = value(values.spotRate2);
        Real spotRate3 = value(values.spotRate3);
        Real couponRate = value(values.couponRate);
        Real faceValue = value(values.faceValue);

        spotRate1.markForgeInputAndDiff();
        spotRate2.markForgeInputAndDiff();
        spotRate3.markForgeInputAndDiff();
        couponRate.markForgeInputAndDiff();
        faceValue.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId spotRate1NodeId = spotRate1.forgeNodeId();
        forge::NodeId spotRate2NodeId = spotRate2.forgeNodeId();
        forge::NodeId spotRate3NodeId = spotRate3.forgeNodeId();
        forge::NodeId couponRateNodeId = couponRate.forgeNodeId();
        forge::NodeId faceValueNodeId = faceValue.forgeNodeId();

        // Build the data struct from the marked inputs
        BondsData data{spotRate1, spotRate2, spotRate3, couponRate, faceValue};

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
        buffer->setValue(spotRate1NodeId, value(values.spotRate1));
        buffer->setValue(spotRate2NodeId, value(values.spotRate2));
        buffer->setValue(spotRate3NodeId, value(values.spotRate3));
        buffer->setValue(couponRateNodeId, value(values.couponRate));
        buffer->setValue(faceValueNodeId, value(values.faceValue));

        // Execute (forward + backward in one call)
        kernel->execute(*buffer);

        // Get the price value
        double priceOutput[1];
        buffer->getLanes(priceNodeId, priceOutput);
        double priceValue = priceOutput[0];

        // Get gradients directly
        int vectorWidth = buffer->getVectorWidth();
        std::vector<size_t> gradientIndices = {
            static_cast<size_t>(spotRate1NodeId) * vectorWidth,
            static_cast<size_t>(spotRate2NodeId) * vectorWidth,
            static_cast<size_t>(spotRate3NodeId) * vectorWidth,
            static_cast<size_t>(couponRateNodeId) * vectorWidth,
            static_cast<size_t>(faceValueNodeId) * vectorWidth
        };
        std::vector<double> gradients(5);
        double* gradOutputs[4] = {gradients.data(), nullptr, nullptr, nullptr};
        buffer->getGradientLanes(gradientIndices, gradOutputs);

        derivatives.spotRate1 = gradients[0];
        derivatives.spotRate2 = gradients[1];
        derivatives.spotRate3 = gradients[2];
        derivatives.couponRate = gradients[3];
        derivatives.faceValue = gradients[4];

        return Real(priceValue);
    }
}

namespace {
    Real priceBonds(const BondsData& value) {
        Date today = Date(15, January, 2015);
        Settings::instance().evaluationDate() = today;

        std::vector<Date> spotDates = {Date(15, January, 2015), Date(15, July, 2015),
                                       Date(15, January, 2016)};
        std::vector<Real> spotRates = {value.spotRate1, value.spotRate2, value.spotRate3};
        DayCounter dayCount = Thirty360(Thirty360::USA);
        Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
        // Using Continuous compounding (Compounded requires patching InterestRate)
        Compounding compounding = Continuous;
        Frequency compoundingFrequency = NoFrequency;

        Handle<YieldTermStructure> spotCurveHandle(ext::make_shared<ZeroCurve>(
            spotDates, spotRates, dayCount, calendar, Linear(), compounding, compoundingFrequency));

        Date issueDate = Date(15, January, 2015);
        Date maturityDate = Date(15, January, 2016);

        Schedule schedule(issueDate, maturityDate, Period(Semiannual), calendar, Unadjusted,
                          Unadjusted, DateGeneration::Backward, false);
        std::vector<Real> coupons = {value.couponRate};

        auto fixedRateBond =
            ext::make_shared<FixedRateBond>(0, value.faceValue, schedule, coupons, dayCount);

        auto bondEngine = ext::make_shared<DiscountingBondEngine>(spotCurveHandle);
        fixedRateBond->setPricingEngine(bondEngine);

        return fixedRateBond->NPV();
    }
}

BOOST_AUTO_TEST_CASE(testBondsDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing bonds derivatives with Forge AAD...");

    // input
    auto data = BondsData{0.0, 0.005, 0.007, 0.06, 100.0};

    // bumping
    auto derivatives_bumping = BondsData{};
    auto expected = priceWithBumping(data, derivatives_bumping, priceBonds);

    // Forge AAD
    auto derivatives_forge = BondsData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceBonds);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bumping.spotRate1, derivatives_forge.spotRate1, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.spotRate2, derivatives_forge.spotRate2, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.spotRate3, derivatives_forge.spotRate3, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.couponRate, derivatives_forge.couponRate, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.faceValue, derivatives_forge.faceValue, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
