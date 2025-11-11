// Test: Validate AAD Greeks against finite difference approximations
// This ensures that Forge automatic differentiation produces accurate derivatives

#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

// Include Forge adapter headers BEFORE QuantLib
#include <ql/forge/config.hpp>
#include <ql/forge/math.hpp>
#include <ql/forge/session.hpp>
#include <ql/forge/traits.hpp>

// Now include QuantLib
#include <ql/quantlib.hpp>

using namespace QuantLib;

// Helper function to compute option price with passive double values
double computePricePassive(double S, double r, double sigma, double q,
                          Date today, Date maturity, double strike) {
    DayCounter dc = Actual365Fixed();
    Calendar cal = TARGET();

    Handle<Quote> spotQ(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> riskFreeTS(
        ext::make_shared<FlatForward>(today, r, dc));
    Handle<YieldTermStructure> divTS(
        ext::make_shared<FlatForward>(today, q, dc));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, cal, sigma, dc));

    ext::shared_ptr<BlackScholesMertonProcess> process =
        ext::make_shared<BlackScholesMertonProcess>(
            spotQ, divTS, riskFreeTS, volTS);

    ext::shared_ptr<StrikedTypePayoff> payoff =
        ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturity);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

    return option.NPV();
}

// Finite difference approximation
double finiteDifference(double S, double r, double sigma, double q,
                       Date today, Date maturity, double strike,
                       int paramIndex, double epsilon) {
    double f0, f1, f2;

    switch(paramIndex) {
        case 0: // Delta: ∂V/∂S
            f0 = computePricePassive(S - epsilon, r, sigma, q, today, maturity, strike);
            f2 = computePricePassive(S + epsilon, r, sigma, q, today, maturity, strike);
            break;
        case 1: // Rho: ∂V/∂r
            f0 = computePricePassive(S, r - epsilon, sigma, q, today, maturity, strike);
            f2 = computePricePassive(S, r + epsilon, sigma, q, today, maturity, strike);
            break;
        case 2: // Vega: ∂V/∂σ
            f0 = computePricePassive(S, r, sigma - epsilon, q, today, maturity, strike);
            f2 = computePricePassive(S, r, sigma + epsilon, q, today, maturity, strike);
            break;
        default:
            throw std::runtime_error("Invalid parameter index");
    }

    // Central difference: (f(x+h) - f(x-h)) / (2h)
    return (f2 - f0) / (2.0 * epsilon);
}

int main() {
    try {
        std::cout << "==============================================\n";
        std::cout << "Test: AAD Greeks vs Finite Differences\n";
        std::cout << "==============================================\n\n";

        // Test parameters
        double S0 = 100.0;
        double r0 = 0.05;
        double sigma0 = 0.20;
        double q0 = 0.02;
        double strike = 100.0;

        Date today = Date::todaysDate();
        Settings::instance().evaluationDate() = today;
        Date maturity = TARGET().advance(today, 1, Years);

        std::cout << "Test configuration:\n";
        std::cout << "  Spot:      " << S0 << "\n";
        std::cout << "  Rate:      " << r0 << "\n";
        std::cout << "  Volatility:" << sigma0 << "\n";
        std::cout << "  Strike:    " << strike << "\n";
        std::cout << "  Maturity:  1 year\n\n";

        // ===== AAD Computation =====
        std::cout << "Computing Greeks with AAD...\n";
        qlforge::Session session;

        Real S = S0;
        Real r = r0;
        Real sigma = sigma0;
        Real q = q0;

        auto h_S = session.markInput(S);
        auto h_r = session.markInput(r);
        auto h_sigma = session.markInput(sigma);

        // Set up option pricing (same as helper function but with active types)
        DayCounter dc = Actual365Fixed();
        Calendar cal = TARGET();

        Handle<Quote> spotQ(ext::make_shared<SimpleQuote>(S));
        Handle<YieldTermStructure> riskFreeTS(
            ext::make_shared<FlatForward>(today, r, dc));
        Handle<YieldTermStructure> divTS(
            ext::make_shared<FlatForward>(today, q, dc));
        Handle<BlackVolTermStructure> volTS(
            ext::make_shared<BlackConstantVol>(today, cal, sigma, dc));

        ext::shared_ptr<BlackScholesMertonProcess> process =
            ext::make_shared<BlackScholesMertonProcess>(
                spotQ, divTS, riskFreeTS, volTS);

        ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        ext::shared_ptr<Exercise> exercise =
            ext::make_shared<EuropeanExercise>(maturity);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

        Real npv = option.NPV();
        auto h_npv = session.markOutput(npv);

        session.finalize();
        session.setInputValue(h_S, S0);
        session.setInputValue(h_r, r0);
        session.setInputValue(h_sigma, sigma0);
        session.execute();

        double price_aad = session.getOutputValue(h_npv);
        double delta_aad = session.getAdjoint(h_S);
        double rho_aad = session.getAdjoint(h_r);
        double vega_aad = session.getAdjoint(h_sigma);

        std::cout << "  Price: " << price_aad << "\n";
        std::cout << "  Delta: " << delta_aad << "\n";
        std::cout << "  Rho:   " << rho_aad << "\n";
        std::cout << "  Vega:  " << vega_aad << "\n\n";

        // ===== Finite Difference Computation =====
        std::cout << "Computing Greeks with finite differences...\n";
        double epsilon = 1e-5;
        double price_fd = computePricePassive(S0, r0, sigma0, q0, today, maturity, strike);
        double delta_fd = finiteDifference(S0, r0, sigma0, q0, today, maturity, strike, 0, epsilon);
        double rho_fd = finiteDifference(S0, r0, sigma0, q0, today, maturity, strike, 1, epsilon);
        double vega_fd = finiteDifference(S0, r0, sigma0, q0, today, maturity, strike, 2, epsilon);

        std::cout << "  Price: " << price_fd << "\n";
        std::cout << "  Delta: " << delta_fd << "\n";
        std::cout << "  Rho:   " << rho_fd << "\n";
        std::cout << "  Vega:  " << vega_fd << "\n\n";

        // ===== Comparison =====
        std::cout << "==============================================\n";
        std::cout << "Comparison (AAD vs FD):\n";
        std::cout << "==============================================\n";
        std::cout << std::fixed << std::setprecision(8);

        double price_diff = std::abs(price_aad - price_fd);
        double delta_diff = std::abs(delta_aad - delta_fd);
        double rho_diff = std::abs(rho_aad - rho_fd);
        double vega_diff = std::abs(vega_aad - vega_fd);

        std::cout << "Price difference: " << price_diff << "\n";
        std::cout << "Delta difference: " << delta_diff << "\n";
        std::cout << "Rho difference:   " << rho_diff << "\n";
        std::cout << "Vega difference:  " << vega_diff << "\n\n";

        // Validation thresholds
        double price_tol = 1e-6;
        double greek_tol = 1e-4;  // Greeks are less precise with FD

        bool price_ok = price_diff < price_tol;
        bool delta_ok = delta_diff < greek_tol;
        bool rho_ok = rho_diff < greek_tol;
        bool vega_ok = vega_diff < greek_tol;

        std::cout << "Validation (tolerance " << greek_tol << "):\n";
        std::cout << "  Price: " << (price_ok ? "PASS" : "FAIL") << "\n";
        std::cout << "  Delta: " << (delta_ok ? "PASS" : "FAIL") << "\n";
        std::cout << "  Rho:   " << (rho_ok ? "PASS" : "FAIL") << "\n";
        std::cout << "  Vega:  " << (vega_ok ? "PASS" : "FAIL") << "\n";
        std::cout << "==============================================\n";

        if (price_ok && delta_ok && rho_ok && vega_ok) {
            std::cout << "\n✓ All tests PASSED!\n";
            return 0;
        } else {
            std::cout << "\n✗ Some tests FAILED!\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
