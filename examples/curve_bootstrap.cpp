// Curve Bootstrapping Sensitivity with Forge AAD
// Demonstrates computation of NPV sensitivities to curve input rates using automatic differentiation

#include <iostream>
#include <iomanip>
#include <vector>

// Include Forge adapter headers BEFORE QuantLib
#include <ql/forge/config.hpp>
#include <ql/forge/math.hpp>
#include <ql/forge/session.hpp>
#include <ql/forge/traits.hpp>

// Now include QuantLib (Real will be forge::fdouble)
#include <ql/quantlib.hpp>

using namespace QuantLib;

int main() {
    try {
        std::cout << "==============================================\n";
        std::cout << "Curve Bootstrapping Sensitivity with Forge AAD\n";
        std::cout << "==============================================\n\n";

        // Create a Forge session for recording
        qlforge::Session session;

        // Setup dates
        Date today = Date::todaysDate();
        Settings::instance().evaluationDate() = today;
        Calendar calendar = TARGET();
        DayCounter dayCounter = Actual360();

        std::cout << "Evaluation date: " << today << "\n\n";

        // Define curve pillars (rates as active variables)
        struct Pillar {
            Period tenor;
            Real rate;
            forge::InputHandle handle;
        };

        std::vector<Pillar> pillars = {
            {1 * Months, 0.05, forge::InputHandle()},
            {3 * Months, 0.051, forge::InputHandle()},
            {6 * Months, 0.053, forge::InputHandle()},
            {1 * Years, 0.055, forge::InputHandle()},
            {2 * Years, 0.058, forge::InputHandle()},
            {5 * Years, 0.06, forge::InputHandle()}
        };

        std::cout << "Curve pillars:\n";
        for (auto& pillar : pillars) {
            // Mark each rate as an independent variable
            pillar.handle = session.markInput(pillar.rate);
            std::cout << "  " << std::setw(10) << pillar.tenor
                      << ": " << std::fixed << std::setprecision(5)
                      << qlforge::to_value(pillar.rate) * 100 << "%\n";
        }
        std::cout << "\n";

        // Build discount curve from deposit quotes
        std::vector<ext::shared_ptr<RateHelper>> instruments;

        for (const auto& pillar : pillars) {
            ext::shared_ptr<Quote> quote = ext::make_shared<SimpleQuote>(pillar.rate);
            Handle<Quote> quoteHandle(quote);

            ext::shared_ptr<RateHelper> helper;
            if (pillar.tenor <= 1 * Years) {
                // Use deposits for short tenors
                helper = ext::make_shared<DepositRateHelper>(
                    quoteHandle,
                    pillar.tenor,
                    2,                    // Settlement days
                    calendar,
                    ModifiedFollowing,
                    false,
                    dayCounter);
            } else {
                // Use swaps for longer tenors (simplified - using fixed rate)
                ext::shared_ptr<IborIndex> index =
                    ext::make_shared<Euribor6M>();
                helper = ext::make_shared<SwapRateHelper>(
                    quoteHandle,
                    pillar.tenor,
                    calendar,
                    Annual,
                    ModifiedFollowing,
                    dayCounter,
                    index);
            }

            instruments.push_back(helper);
        }

        // Bootstrap the curve
        ext::shared_ptr<YieldTermStructure> curve =
            ext::make_shared<PiecewiseYieldCurve<Discount, LogLinear>>(
                today,
                instruments,
                dayCounter);
        curve->enableExtrapolation();

        Handle<YieldTermStructure> curveHandle(curve);

        std::cout << "Bootstrapped curve (discount factors):\n";
        for (const auto& pillar : pillars) {
            Date maturityDate = calendar.advance(today, pillar.tenor);
            Real df = curve->discount(maturityDate);
            std::cout << "  " << std::setw(10) << pillar.tenor
                      << ": " << std::fixed << std::setprecision(8)
                      << qlforge::to_value(df) << "\n";
        }
        std::cout << "\n";

        // Price a simple instrument using the curve
        // For demonstration: a fixed-rate bond
        Real faceAmount = 1000000.0;  // 1M face value
        Real couponRate = 0.055;      // 5.5% coupon
        Date issueDate = today;
        Date maturityDate = calendar.advance(today, 3 * Years);

        Schedule schedule(issueDate, maturityDate,
                         Period(Annual),
                         calendar,
                         ModifiedFollowing,
                         ModifiedFollowing,
                         DateGeneration::Backward,
                         false);

        FixedRateBond bond(
            0,                              // Settlement days
            faceAmount,
            schedule,
            std::vector<Rate>(1, couponRate),
            dayCounter,
            ModifiedFollowing,
            100.0,                          // Redemption
            issueDate);

        // Set the pricing engine using our bootstrapped curve
        ext::shared_ptr<PricingEngine> bondEngine =
            ext::make_shared<DiscountingBondEngine>(curveHandle);
        bond.setPricingEngine(bondEngine);

        // Calculate NPV (this records the computation graph)
        Real npv = bond.NPV();

        std::cout << "Bond details:\n";
        std::cout << "  Face amount:     " << faceAmount << "\n";
        std::cout << "  Coupon rate:     " << couponRate * 100 << "%\n";
        std::cout << "  Maturity:        " << maturityDate << "\n";
        std::cout << "  Clean price:     " << std::fixed << std::setprecision(2)
                  << qlforge::to_value(bond.cleanPrice()) << "\n";
        std::cout << "  Dirty price:     " << qlforge::to_value(bond.dirtyPrice()) << "\n\n";

        // Mark output
        auto h_npv = session.markOutput(npv);

        std::cout << "Compiling computation graph...\n";
        session.finalize();
        std::cout << "Compilation complete.\n\n";

        // Set input values and execute
        std::cout << "Executing forward and adjoint passes...\n";
        for (const auto& pillar : pillars) {
            session.setInputValue(pillar.handle, qlforge::to_value(pillar.rate));
        }
        session.execute();
        std::cout << "Execution complete.\n\n";

        // Extract results
        double bondNPV = session.getOutputValue(h_npv);

        std::cout << "==============================================\n";
        std::cout << "Results:\n";
        std::cout << "==============================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Bond NPV:        " << bondNPV << "\n\n";

        std::cout << "Sensitivities (∂NPV/∂rate) - Curve Risk:\n";
        std::cout << std::setw(12) << "Tenor"
                  << std::setw(15) << "Sensitivity"
                  << std::setw(15) << "1bp Impact\n";
        std::cout << std::string(42, '-') << "\n";

        for (const auto& pillar : pillars) {
            double sensitivity = session.getAdjoint(pillar.handle);
            double bpv = sensitivity * 0.0001;  // 1bp = 0.01%

            std::cout << std::setw(12) << pillar.tenor
                      << std::setw(15) << std::fixed << std::setprecision(2) << sensitivity
                      << std::setw(15) << std::fixed << std::setprecision(2) << bpv << "\n";
        }

        std::cout << "==============================================\n\n";

        std::cout << "Interpretation:\n";
        std::cout << "- Positive sensitivity: Bond value increases when rate increases\n";
        std::cout << "  (usually negative for bonds, positive for rate products)\n";
        std::cout << "- 1bp Impact: Change in NPV for a 1 basis point (0.01%) rate increase\n";
        std::cout << "- Larger absolute values indicate higher sensitivity to that pillar\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
