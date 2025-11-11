/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2005, 2006, 2007, 2009 StatPro Italia srl
 Copyright (C) 2024 Forge contributors

 This file is part of QuantLib / Forge integration module.
 It is adapted from the XAD/QuantLib-Risks example to demonstrate
 equivalent functionality with Forge AAD engine.

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*
This example shows how to price a portfolio of European equity options with
sensitivities computed via Forge AAD (or bumping if QLFORGE_DISABLE_AAD is ON).
It measures performance and reports the times.

This sample validates Forge performance with an analytic engine, comparing
with the XAD implementation.
*/

#include <ql/qldefines.hpp>
#if !defined(BOOST_ALL_NO_LIB) && defined(BOOST_MSVC)
#    include <ql/auto_link.hpp>
#endif

// Include Forge adapter headers BEFORE QuantLib
#include <ql/forge/config.hpp>
#include <ql/forge/math.hpp>
#include <ql/forge/session.hpp>
#include <ql/forge/traits.hpp>

// Now include QuantLib
#include <ql/exercise.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

// to record all sensitivities of the portfolio
struct OptionSensitivities {
    std::vector<double> rhos;
    std::vector<double> strikeSensitivities;
    std::vector<double> deltas;
    std::vector<double> vegas;
    double dividendRho;
};

Real priceEuropean(const std::vector<Date>& dates,
                   const std::vector<Rate>& rates,
                   const std::vector<Real>& vols,
                   const Calendar& calendar,
                   const Date maturity,
                   const std::vector<Real>& strikes,
                   const Date settlementDate,
                   DayCounter& dayCounter,
                   Date todaysDate,
                   Spread dividendYield,
                   Option::Type type,
                   const std::vector<Real>& underlyings) {

    auto europeanExercise = ext::make_shared<EuropeanExercise>(maturity);

    // setup the yield/dividend/vol curves
    Handle<YieldTermStructure> termStructure(ext::make_shared<ZeroCurve>(dates, rates, dayCounter));
    Handle<YieldTermStructure> flatDividendTS(
        ext::make_shared<FlatForward>(settlementDate, dividendYield, dayCounter));
    // cut first date (settlement date) for vol curve
    auto dvol = std::vector<Date>(dates.begin() + 1, dates.end());
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackVarianceCurve>(settlementDate, dvol, vols, dayCounter));

    Real value = 0.0;
    // portfolio with several underlying values
    for (auto& underlying : underlyings) {
        Handle<Quote> underlyingH(ext::make_shared<SimpleQuote>(underlying));
        auto bsmProcess = ext::make_shared<BlackScholesMertonProcess>(underlyingH, flatDividendTS,
                                                                      termStructure, volTS);
        auto engine = ext::make_shared<AnalyticEuropeanEngine>(bsmProcess);

        // and options with several strikes for each underlying value
        for (auto& strike : strikes) {
            auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);

            // options
            auto european = ext::make_shared<VanillaOption>(payoff, europeanExercise);
            // computing the option price with the analytic Black-Scholes formulae
            european->setPricingEngine(engine);

            value += european->NPV();
        }
    }
    return value;
}

// price with Forge sensitivities
#ifndef QLFORGE_DISABLE_AAD

Real priceWithSensi(const std::vector<Date>& dates,
                    const std::vector<Rate>& rates,
                    const std::vector<Real>& vols,
                    const Calendar& calendar,
                    const Date& maturity,
                    const std::vector<Real>& strikes,
                    const Date settlementDate,
                    DayCounter& dayCounter,
                    Date todaysDate,
                    Spread dividendYield,
                    Option::Type type,
                    const std::vector<Real>& underlyings,
                    OptionSensitivities& sensiOutput) {

    // Create Forge session
    qlforge::Session session;

    // Copy inputs and mark them as independent variables
    auto t_rates = rates;
    auto t_vols = vols;
    auto t_strikes = strikes;
    auto t_underlyings = underlyings;
    Spread t_dividendYield = dividendYield;

    std::vector<qlforge::InputHandle> rate_handles;
    for (auto& r : t_rates) {
        rate_handles.push_back(session.markInput(r));
    }
    std::vector<qlforge::InputHandle> vol_handles;
    for (auto& v : t_vols) {
        vol_handles.push_back(session.markInput(v));
    }
    std::vector<qlforge::InputHandle> strike_handles;
    for (auto& s : t_strikes) {
        strike_handles.push_back(session.markInput(s));
    }
    std::vector<qlforge::InputHandle> underlying_handles;
    for (auto& u : t_underlyings) {
        underlying_handles.push_back(session.markInput(u));
    }
    auto dividend_handle = session.markInput(t_dividendYield);

    // price
    Real value =
        priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                      dayCounter, todaysDate, t_dividendYield, type, t_underlyings);

    // Mark output
    auto value_handle = session.markOutput(value);

    // Finalize and execute
    session.finalize();

    // Set input values
    for (size_t i = 0; i < t_rates.size(); ++i) {
        session.setInputValue(rate_handles[i], qlforge::to_value(rates[i]));
    }
    for (size_t i = 0; i < t_vols.size(); ++i) {
        session.setInputValue(vol_handles[i], qlforge::to_value(vols[i]));
    }
    for (size_t i = 0; i < t_strikes.size(); ++i) {
        session.setInputValue(strike_handles[i], qlforge::to_value(strikes[i]));
    }
    for (size_t i = 0; i < t_underlyings.size(); ++i) {
        session.setInputValue(underlying_handles[i], qlforge::to_value(underlyings[i]));
    }
    session.setInputValue(dividend_handle, qlforge::to_value(dividendYield));

    // Execute forward and adjoint passes
    session.execute();

    // Extract sensitivities
    sensiOutput.rhos.clear();
    sensiOutput.rhos.reserve(t_rates.size());
    for (size_t i = 0; i < t_rates.size(); ++i) {
        sensiOutput.rhos.push_back(session.getAdjoint(rate_handles[i]));
    }
    sensiOutput.vegas.clear();
    sensiOutput.vegas.reserve(t_vols.size());
    for (size_t i = 0; i < t_vols.size(); ++i) {
        sensiOutput.vegas.push_back(session.getAdjoint(vol_handles[i]));
    }
    sensiOutput.strikeSensitivities.clear();
    sensiOutput.strikeSensitivities.reserve(t_strikes.size());
    for (size_t i = 0; i < t_strikes.size(); ++i) {
        sensiOutput.strikeSensitivities.push_back(session.getAdjoint(strike_handles[i]));
    }
    sensiOutput.deltas.clear();
    sensiOutput.deltas.reserve(t_underlyings.size());
    for (size_t i = 0; i < t_underlyings.size(); ++i) {
        sensiOutput.deltas.push_back(session.getAdjoint(underlying_handles[i]));
    }
    sensiOutput.dividendRho = session.getAdjoint(dividend_handle);

    return session.getOutputValue(value_handle);
}

#else // pricing with bumping, as we don't have Forge available here

Real priceWithSensi(const std::vector<Date>& dates,
                    const std::vector<Rate>& rates,
                    const std::vector<Real>& vols,
                    const Calendar& calendar,
                    const Date& maturity,
                    const std::vector<Real>& strikes,
                    const Date settlementDate,
                    DayCounter& dayCounter,
                    Date todaysDate,
                    Spread dividendYield,
                    Option::Type type,
                    const std::vector<Real>& underlyings,
                    OptionSensitivities& sensiOutput) {

    // copy inputs for bumping
    auto t_rates = rates;
    auto t_vols = vols;
    auto t_strikes = strikes;

    Real value =
        priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                      dayCounter, todaysDate, dividendYield, type, underlyings);

    // bump each input independently and re-run the pricer to see its impact
    Real eps = 1e-5;
    sensiOutput.rhos.clear();
    sensiOutput.rhos.reserve(t_rates.size());
    for (Size i = 0; i < t_rates.size(); ++i) {
        t_rates[i] += eps;
        Real v1 =
            priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                          dayCounter, todaysDate, dividendYield, type, underlyings);
        sensiOutput.rhos.push_back((v1 - value) / eps);
        t_rates[i] -= eps;
    }

    sensiOutput.vegas.clear();
    sensiOutput.vegas.reserve(t_vols.size());
    for (Size i = 0; i < t_vols.size(); ++i) {
        t_vols[i] += eps;
        Real v1 =
            priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                          dayCounter, todaysDate, dividendYield, type, underlyings);
        sensiOutput.vegas.push_back((v1 - value) / eps);
        t_vols[i] -= eps;
    }

    sensiOutput.strikeSensitivities.clear();
    sensiOutput.strikeSensitivities.reserve(t_strikes.size());
    for (Size i = 0; i < t_strikes.size(); ++i) {
        t_strikes[i] += eps;
        Real v1 =
            priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                          dayCounter, todaysDate, dividendYield, type, underlyings);
        sensiOutput.strikeSensitivities.push_back((v1 - value) / eps);
        t_strikes[i] -= eps;
    }

    auto t_underlyings = underlyings;
    sensiOutput.deltas.clear();
    sensiOutput.deltas.reserve(underlyings.size());
    for (Size i = 0; i < t_underlyings.size(); ++i) {
        t_underlyings[i] += eps;
        Real v1 =
            priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                          dayCounter, todaysDate, dividendYield, type, t_underlyings);
        sensiOutput.deltas.push_back((v1 - value) / eps);
        t_underlyings[i] -= eps;
    }

    Real v1 = priceEuropean(dates, t_rates, t_vols, calendar, maturity, t_strikes, settlementDate,
                            dayCounter, todaysDate, dividendYield + eps, type, underlyings);
    sensiOutput.dividendRho = (v1 - value) / eps;

    return value;
}

#endif

void printResults(double v, const OptionSensitivities& sensiOutput) {
    std::cout << "\nGreeks:\n";
    std::cout << "Rhos                  = [";
    for (auto& rho : sensiOutput.rhos) {
        std::cout << rho << ", ";
    }
    std::cout << "]\n";
    std::cout << "Strike Sensitivities = [";
    for (auto& s : sensiOutput.strikeSensitivities) {
        std::cout << s << ", ";
    }
    std::cout << "]\n";
    std::cout << "Vegas               = [";
    for (auto& vega : sensiOutput.vegas) {
        std::cout << vega << ", ";
    }
    std::cout << "]\n";
    std::cout << "Deltas               = [";
    for (auto& delta : sensiOutput.deltas) {
        std::cout << delta << ", ";
    }
    std::cout << "]\n";
    std::cout << "Dividend Rho         = " << sensiOutput.dividendRho << "\n";

    std::cout << std::endl;
}

int main() {
    try {
        // set up dates
        Calendar calendar = TARGET();
        Date todaysDate(15, May, 1998);
        Date settlementDate(17, May, 1998);
        Settings::instance().evaluationDate() = todaysDate;

        // curves
        std::vector<Integer> t = {13, 41, 75, 165, 256, 345, 524, 703};
        std::vector<Rate> r = {0.0357, 0.0349, 0.0341, 0.0355, 0.0359, 0.0368, 0.0386, 0.0401};
        std::vector<Volatility> vols = {0.20, 0.18, 0.178, 0.183, 0.192, 0.203, 0.215, 0.208};

        std::vector<Date> dates;
        std::vector<Rate> rates;
        dates.push_back(settlementDate);
        rates.push_back(r[0]);
        Size i;
        for (i = 0; i < t.size(); ++i) {
            dates.push_back(settlementDate + t[i]);
            rates.push_back(r[i]);
        }

        // our options
        Option::Type type(Option::Put);
        std::vector<Real> underlyings = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60};
        std::vector<Real> strikes;
        for (Real s = 10.; s < 80.; s += 1.) {
            strikes.push_back(s);
        }
        Spread dividendYield = 0.01;
        Date maturity(17, May, 1999);
        DayCounter dayCounter = Actual365Fixed();

        std::cout.precision(5);
        constexpr int N = 100;

        // pricing without sensitivities
        std::cout << "Pricing european equity option portfolio without sensitivities..\n";
        double v = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            v = qlforge::to_value(priceEuropean(dates, rates, vols, calendar, maturity, strikes, settlementDate,
                              dayCounter, todaysDate, dividendYield, type, underlyings));
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto time_plain =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) *
            1e-3 / N;
        std::cout << "Portfolio value: " << v << "\n";

        // pricing with sensitivities
        std::vector<Real> gradient;
        std::cout << "Pricing european equity option portfolio with sensitivities...\n";
        OptionSensitivities sensi;
        double v2 = 0.0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            v2 = priceWithSensi(dates, rates, vols, calendar, maturity, strikes, settlementDate,
                                dayCounter, todaysDate, dividendYield, type, underlyings, sensi);
        }
        end = std::chrono::high_resolution_clock::now();
        auto time_sensi =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) *
            1e-3 / N;
        std::cout << "Portfolio value: " << v2 << "\n";

        printResults(v2, sensi);

        std::cout << "Plain time : " << time_plain << "ms\n"
                  << "Sensi time : " << time_sensi << "ms\n"
                  << "Factor     : " << time_sensi / time_plain << "x\n";

        return 0;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        return 1;
    }
}
