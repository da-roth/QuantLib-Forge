// Barrier Option Pricing with Forge AAD
// Demonstrates computation of barrier option price and Greeks using automatic differentiation
// Note: Discontinuities at the barrier level may affect derivative accuracy

#include <iostream>
#include <iomanip>

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
        std::cout << "Barrier Option Pricing with Forge AAD\n";
        std::cout << "==============================================\n\n";

        // Create a Forge session for recording
        qlforge::Session session;

        // Market data as forge::fdouble (because Real = forge::fdouble)
        Real spotPrice = 95.0;
        Real riskFreeRate = 0.08;
        Real dividendYield = 0.04;
        Real volatility = 0.25;
        Real barrier = 100.0;        // Up-and-out barrier
        Real rebate = 3.0;           // Rebate if barrier is hit

        // Mark independent variables for differentiation
        auto h_spot = session.markInput(spotPrice);
        auto h_rate = session.markInput(riskFreeRate);
        auto h_vol = session.markInput(volatility);

        std::cout << "Input parameters:\n";
        std::cout << "  Spot price:      " << qlforge::to_value(spotPrice) << "\n";
        std::cout << "  Strike:          90.0\n";
        std::cout << "  Barrier:         " << qlforge::to_value(barrier) << " (Up-and-Out)\n";
        std::cout << "  Rebate:          " << qlforge::to_value(rebate) << "\n";
        std::cout << "  Risk-free rate:  " << qlforge::to_value(riskFreeRate) << "\n";
        std::cout << "  Dividend yield:  " << qlforge::to_value(dividendYield) << "\n";
        std::cout << "  Volatility:      " << qlforge::to_value(volatility) << "\n\n";

        // Set up the option
        Date today = Date::todaysDate();
        Settings::instance().evaluationDate() = today;

        DayCounter dayCounter = Actual365Fixed();
        Calendar calendar = TARGET();

        Date maturity = calendar.advance(today, 6, Months);
        Real strike = 90.0;

        // Create term structures
        Handle<Quote> spotQuote(ext::make_shared<SimpleQuote>(spotPrice));
        Handle<YieldTermStructure> riskFreeTS(
            ext::make_shared<FlatForward>(today, riskFreeRate, dayCounter));
        Handle<YieldTermStructure> dividendTS(
            ext::make_shared<FlatForward>(today, dividendYield, dayCounter));
        Handle<BlackVolTermStructure> volTS(
            ext::make_shared<BlackConstantVol>(today, calendar, volatility, dayCounter));

        // Black-Scholes-Merton process
        ext::shared_ptr<BlackScholesMertonProcess> bsmProcess =
            ext::make_shared<BlackScholesMertonProcess>(
                spotQuote, dividendTS, riskFreeTS, volTS);

        // Create the barrier option
        // Up-and-out call: knock-out when spot hits barrier from below
        ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        ext::shared_ptr<Exercise> europeanExercise =
            ext::make_shared<EuropeanExercise>(maturity);

        BarrierOption barrierOption(
            Barrier::UpOut,      // Knock-out when spot goes up through barrier
            barrier,             // Barrier level
            rebate,              // Rebate paid if knocked out
            payoff,
            europeanExercise);

        // Set pricing engine (analytical formula for European barrier options)
        ext::shared_ptr<PricingEngine> engine =
            ext::make_shared<AnalyticBarrierEngine>(bsmProcess);
        barrierOption.setPricingEngine(engine);

        // Calculate NPV (this records the computation graph)
        Real npv = barrierOption.NPV();

        std::cout << "Option details:\n";
        std::cout << "  Type:            European Call with Up-and-Out Barrier\n";
        std::cout << "  Strike:          " << strike << "\n";
        std::cout << "  Barrier:         " << barrier << " (knock-out above)\n";
        std::cout << "  Rebate:          " << rebate << "\n";
        std::cout << "  Maturity:        " << maturity << "\n";
        std::cout << "  Time to expiry:  " << dayCounter.yearFraction(today, maturity) << " years\n\n";

        // Mark output
        auto h_npv = session.markOutput(npv);

        std::cout << "Compiling computation graph...\n";
        session.finalize();
        std::cout << "Compilation complete.\n\n";

        // Set input values and execute
        std::cout << "Executing forward and adjoint passes...\n";
        session.setInputValue(h_spot, 95.0);
        session.setInputValue(h_rate, 0.08);
        session.setInputValue(h_vol, 0.25);
        session.execute();
        std::cout << "Execution complete.\n\n";

        // Extract results
        double price = session.getOutputValue(h_npv);
        double delta = session.getAdjoint(h_spot);    // ∂NPV/∂S
        double rho = session.getAdjoint(h_rate);      // ∂NPV/∂r
        double vega = session.getAdjoint(h_vol);      // ∂NPV/∂σ

        // Print results
        std::cout << "==============================================\n";
        std::cout << "Results:\n";
        std::cout << "==============================================\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Option Price:    " << price << "\n";
        std::cout << "\nGreeks (via AAD):\n";
        std::cout << "  Delta (∂V/∂S): " << delta << "\n";
        std::cout << "  Rho   (∂V/∂r): " << rho << "\n";
        std::cout << "  Vega  (∂V/∂σ): " << vega << "\n";
        std::cout << "==============================================\n\n";

        // Compare with QuantLib's analytical Greeks (if available)
        try {
            std::cout << "QuantLib analytical Greeks (for comparison):\n";
            std::cout << "  Delta:         " << barrierOption.delta() << "\n";
            std::cout << "  Rho:           " << barrierOption.rho() << "\n";
            std::cout << "  Vega:          " << barrierOption.vega() << "\n";
        } catch (const std::exception& e) {
            std::cout << "  (Analytical Greeks not available: " << e.what() << ")\n";
        }

        std::cout << "\n";
        std::cout << "Note: Barrier options have discontinuities at the barrier level,\n";
        std::cout << "which can affect derivative accuracy near the barrier.\n";
        std::cout << "For spot = " << qlforge::to_value(spotPrice)
                  << " and barrier = " << qlforge::to_value(barrier)
                  << ", we are " << (qlforge::to_value(barrier) - qlforge::to_value(spotPrice))
                  << " units away from the barrier.\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
