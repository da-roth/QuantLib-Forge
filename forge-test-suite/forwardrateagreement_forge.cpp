/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Forward Rate Agreement AAD test using Forge.

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
#include <ql/indexes/ibor/usdlibor.hpp>
#include <ql/instruments/forwardrateagreement.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/period.hpp>

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

BOOST_AUTO_TEST_SUITE(ForwardRateAgreementForgeTests)

namespace {

    struct ForwardRateAgreementData {
        Real nominal;
        Real spotRate1;
        Real spotRate2;
        Real spotRate3;
        Real rate;
    };

}

namespace {

    template <class PriceFunc>
    Real priceWithBumping(const ForwardRateAgreementData& value,
                          ForwardRateAgreementData& derivatives,
                          PriceFunc func) {
        // Bumping
        auto eps = 1e-7;
        auto data = value;
        auto v = func(data);

        data.nominal += 1;
        auto vplus = func(data);
        derivatives.nominal = (vplus - v) / 1;
        data = value;

        data.spotRate1 += eps;
        vplus = func(data);
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

        data.rate += eps;
        vplus = func(data);
        derivatives.rate = (vplus - v) / eps;

        return v;
    }

    template <class PriceFunc>
    Real priceWithForgeAAD(const ForwardRateAgreementData& values,
                           ForwardRateAgreementData& derivatives,
                           PriceFunc func) {
        // Forge AAD - Graph recording and JIT compilation
        forge::GraphRecorder recorder;
        recorder.start();

        // Create inputs and mark them for differentiation
        auto data = values;
        data.nominal.markForgeInputAndDiff();
        data.spotRate1.markForgeInputAndDiff();
        data.spotRate2.markForgeInputAndDiff();
        data.spotRate3.markForgeInputAndDiff();
        data.rate.markForgeInputAndDiff();

        // Store node IDs for gradient retrieval
        forge::NodeId nominalNodeId = data.nominal.forgeNodeId();
        forge::NodeId spotRate1NodeId = data.spotRate1.forgeNodeId();
        forge::NodeId spotRate2NodeId = data.spotRate2.forgeNodeId();
        forge::NodeId spotRate3NodeId = data.spotRate3.forgeNodeId();
        forge::NodeId rateNodeId = data.rate.forgeNodeId();

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
        buffer->setValue(spotRate1NodeId, value(values.spotRate1));
        buffer->setValue(spotRate2NodeId, value(values.spotRate2));
        buffer->setValue(spotRate3NodeId, value(values.spotRate3));
        buffer->setValue(rateNodeId, value(values.rate));

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
            static_cast<size_t>(spotRate1NodeId) * vectorWidth,
            static_cast<size_t>(spotRate2NodeId) * vectorWidth,
            static_cast<size_t>(spotRate3NodeId) * vectorWidth,
            static_cast<size_t>(rateNodeId) * vectorWidth
        };
        std::vector<double> gradients(5);
        double* gradOutputs[4] = {gradients.data(), nullptr, nullptr, nullptr};
        buffer->getGradientLanes(gradientIndices, gradOutputs);

        derivatives.nominal = gradients[0];
        derivatives.spotRate1 = gradients[1];
        derivatives.spotRate2 = gradients[2];
        derivatives.spotRate3 = gradients[3];
        derivatives.rate = gradients[4];

        return Real(priceValue);
    }
}

namespace {
    Real priceForwardRateAgreement(const ForwardRateAgreementData& value) {
        Date today(30, June, 2020);
        Settings::instance().evaluationDate() = today;

        std::vector<Date> spotDates;
        spotDates.push_back(Date(30, June, 2020));
        spotDates.push_back(Date(31, December, 2020));
        spotDates.push_back(Date(30, June, 2021));
        std::vector<Real> spotRates;
        spotRates.push_back(value.spotRate1);
        spotRates.push_back(value.spotRate2);
        spotRates.push_back(value.spotRate3);

        DayCounter dayConvention = Actual360();
        Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
        Date startDate = calendar.advance(today, Period(3, Months));
        Date maturityDate = calendar.advance(startDate, Period(3, Months));

        Compounding compounding = Simple;
        Frequency compoundingFrequency = Annual;

        Handle<YieldTermStructure> spotCurve(
            ext::make_shared<ZeroCurve>(spotDates, spotRates, dayConvention, calendar, Linear(),
                                        compounding, compoundingFrequency));
        spotCurve->enableExtrapolation();

        auto index = ext::make_shared<USDLibor>(Period(3, Months), spotCurve);
        index->addFixing(Date(26, June, 2020), 0.05);

        ForwardRateAgreement fra(index, startDate, maturityDate, Position::Long, value.rate,
                                 value.nominal, spotCurve);

        return fra.NPV();
    }
}

BOOST_AUTO_TEST_CASE(testForwardRateAgreementDerivatives) {

    SavedSettings save;
    BOOST_TEST_MESSAGE("Testing forward rate agreement derivatives with Forge AAD...");

    // input
    auto data = ForwardRateAgreementData{100000, 0.5, 0.5, 0.5, 0.06};

    // bumping
    auto derivatives_bumping = ForwardRateAgreementData{};
    auto expected = priceWithBumping(data, derivatives_bumping, priceForwardRateAgreement);

    // Forge AAD
    auto derivatives_forge = ForwardRateAgreementData{};
    auto actual = priceWithForgeAAD(data, derivatives_forge, priceForwardRateAgreement);

    // compare
    QL_CHECK_CLOSE(expected, actual, 1e-9);
    QL_CHECK_CLOSE(derivatives_bumping.nominal, derivatives_forge.nominal, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.spotRate1, derivatives_forge.spotRate1, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.spotRate2, derivatives_forge.spotRate2, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.spotRate3, derivatives_forge.spotRate3, 1e-3);
    QL_CHECK_CLOSE(derivatives_bumping.rate, derivatives_forge.rate, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
