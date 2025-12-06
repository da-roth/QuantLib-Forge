/*******************************************************************************

   Minimal CDF Kernel Reuse Test

   This file tests kernel reuse for the CumulativeNormalDistribution.

   Approach:
   - Use the ORIGINAL CDF (with double) to compute expected/reference values
   - Use the FORGE CDF (with AReal + ABool::If) to build and re-evaluate the kernel

   The Forge CDF should produce correct results on kernel reuse because all
   branches are recorded in the computation graph via ABool::If.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#include "cumulative_normal_distribution.hpp"
#include "cumulative_normal_distribution_forge.hpp"

#include <expressions/Literals.hpp>
#include <expressions/ExpressionTemplates/BinaryOperators.hpp>
#include <expressions/ExpressionTemplates/UnaryOperators.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

using Real = forge::expr::AReal<double>;

// Original CDF with double - used to compute expected values
qlforge_test::CumulativeNormalDistribution<double> originalCDF;

// Test result tracking
struct TestResult {
    std::string name;
    int total = 0;
    int passed = 0;

    void record(bool ok) {
        total++;
        if (ok) passed++;
    }

    bool allPassed() const { return passed == total; }
};

//=============================================================================
// Test Forge CDF kernel reuse
// Build kernel with first input, re-evaluate with different inputs
// Compare against original CDF for expected values
//=============================================================================
TestResult testForgeCDFKernelReuse() {
    TestResult result{"Forge CDF Kernel Reuse"};

    std::cout << "\n=== Testing Forge CDF Kernel Reuse ===\n";
    std::cout << "  Build kernel once, re-evaluate with different inputs\n";
    std::cout << "  Expected values from original CDF (with double)\n\n";

    // Test inputs that span different branches of the error function
    // Branch boundaries in erf: 0.84375, 1.25, 2.857, 6.0
    // After CDF scaling (x * 1/sqrt(2)), these map to different x values
    std::vector<double> testInputs = {
        0.5,    // Small x
        1.0,    // Medium x
        2.0,    // Intermediate x
        5.0,    // Large x (but < 6 after scaling)
        10.0,   // Very large x
        -0.5,   // Negative small
        -2.0,   // Negative intermediate
    };

    // Build kernel with first input using Forge CDF
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = testInputs[0];
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    // Use Forge-aware CDF for kernel building
    qlforge_test::CumulativeNormalDistributionForge<Real> forgeCDF;
    Real y = forgeCDF(x);
    y.markForgeOutput();
    forge::NodeId yId = y.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    std::cout << "  Graph has " << graph.nodes.size() << " nodes\n";

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    std::cout << "  Kernel compiled, testing with " << testInputs.size() << " inputs...\n\n";

    // Test each input
    for (size_t i = 0; i < testInputs.size(); i++) {
        double input = testInputs[i];

        // Get expected value from ORIGINAL CDF (with double)
        double expected = originalCDF(input);

        // Re-evaluate Forge kernel with new input
        double inputVal[4] = {input, input, input, input};
        buffer->setLanes(xId, inputVal);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double outputVal[4];
        buffer->getLanes(yId, outputVal);
        double forgeResult = outputVal[0];

        double relError = std::abs(expected) > 1e-10
            ? std::abs(forgeResult - expected) / std::abs(expected)
            : std::abs(forgeResult - expected);

        bool valueOk = relError < 0.01;  // 1% tolerance
        result.record(valueOk);

        std::cout << "  Input " << std::setw(2) << i + 1 << ": x=" << std::setw(6) << input
                  << " | Expected: " << std::setw(10) << std::fixed << std::setprecision(6) << expected
                  << " | Forge: " << std::setw(10) << forgeResult
                  << " | Error: " << std::setw(8) << std::scientific << std::setprecision(2) << relError
                  << " | " << (valueOk ? "PASS" : "FAIL")
                  << (i == 0 ? " (build)" : " (reuse)")
                  << "\n";
    }

    return result;
}

//=============================================================================
// Test B4-like scenario: build with large x, re-eval with moderate x
//=============================================================================
TestResult testB4Scenario() {
    TestResult result{"B4-like Scenario"};

    std::cout << "\n=== Testing B4-like Scenario ===\n";
    std::cout << "  Build with large x (CDF~1), then re-eval with moderate x\n";
    std::cout << "  This is the scenario that fails with original C++ if/else\n\n";

    // Mimics B4 test failure:
    // Build with very large x (CDF returns ~1.0)
    // Re-eval with moderate x (CDF should return ~0.7)
    std::vector<double> testInputs = {
        10.0,   // Very large (build) - CDF ~ 1.0
        8.0,    // Still large - CDF ~ 1.0
        5.0,    // Large - CDF ~ 0.999999...
        2.0,    // Moderate - CDF ~ 0.977
        0.6,    // Small-ish - CDF ~ 0.726 (like B4 Input Set 5)
        0.0,    // Zero - CDF = 0.5
        -1.0,   // Negative - CDF ~ 0.159
    };

    // Build kernel with first input (large x)
    forge::GraphRecorder recorder;
    recorder.start();

    Real x = testInputs[0];
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    qlforge_test::CumulativeNormalDistributionForge<Real> forgeCDF;
    Real y = forgeCDF(x);
    y.markForgeOutput();
    forge::NodeId yId = y.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    std::cout << "  Graph has " << graph.nodes.size() << " nodes\n";

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    std::cout << "  Kernel compiled, testing...\n\n";

    for (size_t i = 0; i < testInputs.size(); i++) {
        double input = testInputs[i];

        // Expected from original CDF
        double expected = originalCDF(input);

        double inputVal[4] = {input, input, input, input};
        buffer->setLanes(xId, inputVal);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double outputVal[4];
        buffer->getLanes(yId, outputVal);
        double forgeResult = outputVal[0];

        double relError = std::abs(expected) > 1e-10
            ? std::abs(forgeResult - expected) / std::abs(expected)
            : std::abs(forgeResult - expected);

        bool valueOk = relError < 0.01;
        result.record(valueOk);

        std::cout << "  x=" << std::setw(6) << input
                  << " | Expected: " << std::setw(10) << std::fixed << std::setprecision(6) << expected
                  << " | Forge: " << std::setw(10) << forgeResult
                  << " | Error: " << std::setw(8) << std::scientific << std::setprecision(2) << relError
                  << " | " << (valueOk ? "PASS" : "FAIL")
                  << (i == 0 ? " (build)" : "")
                  << "\n";
    }

    return result;
}

//=============================================================================
// Test building with small x, re-eval with large x (opposite direction)
//=============================================================================
TestResult testReverseScenario() {
    TestResult result{"Reverse Scenario (small->large)"};

    std::cout << "\n=== Testing Reverse Scenario ===\n";
    std::cout << "  Build with small x, then re-eval with large x\n\n";

    std::vector<double> testInputs = {
        0.3,    // Small (build)
        0.5,    // Still small
        1.0,    // Medium
        2.0,    // Intermediate
        5.0,    // Large
        10.0,   // Very large
    };

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = testInputs[0];
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    qlforge_test::CumulativeNormalDistributionForge<Real> forgeCDF;
    Real y = forgeCDF(x);
    y.markForgeOutput();
    forge::NodeId yId = y.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    std::cout << "  Graph has " << graph.nodes.size() << " nodes\n";

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    std::cout << "  Kernel compiled, testing...\n\n";

    for (size_t i = 0; i < testInputs.size(); i++) {
        double input = testInputs[i];
        double expected = originalCDF(input);

        double inputVal[4] = {input, input, input, input};
        buffer->setLanes(xId, inputVal);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double outputVal[4];
        buffer->getLanes(yId, outputVal);
        double forgeResult = outputVal[0];

        double relError = std::abs(expected) > 1e-10
            ? std::abs(forgeResult - expected) / std::abs(expected)
            : std::abs(forgeResult - expected);

        bool valueOk = relError < 0.01;
        result.record(valueOk);

        std::cout << "  x=" << std::setw(6) << input
                  << " | Expected: " << std::setw(10) << std::fixed << std::setprecision(6) << expected
                  << " | Forge: " << std::setw(10) << forgeResult
                  << " | Error: " << std::setw(8) << std::scientific << std::setprecision(2) << relError
                  << " | " << (valueOk ? "PASS" : "FAIL")
                  << (i == 0 ? " (build)" : "")
                  << "\n";
    }

    return result;
}

//=============================================================================
// Test C-formula scenario: build with very large negative x, re-eval with small positive x
// This is the exact scenario that breaks C5/C6/C7 in the barrier option tests
//=============================================================================
TestResult testCFormulaScenario() {
    TestResult result{"C-Formula Scenario (large negative -> small positive)"};

    std::cout << "\n=== Testing C-Formula Scenario ===\n";
    std::cout << "  Build with very large NEGATIVE x (like C formula y1 ~ -136)\n";
    std::cout << "  Re-eval with small POSITIVE x (like C formula y1 ~ +0.95)\n";
    std::cout << "  This is the exact scenario that breaks C5/C6/C7 tests!\n\n";

    // These values come from the C8 test output:
    // Input Set 1: y1 = -136 (build)
    // Input Set 5: y1 = +0.95 (re-eval that fails)
    std::vector<double> testInputs = {
        -136.0,   // Very large negative (build) - CDF ~ 0
        -106.0,   // Large negative - CDF ~ 0
        -75.0,    // Large negative - CDF ~ 0
        -57.0,    // Large negative - CDF ~ 0
        -10.0,    // Moderate negative - CDF ~ 0
        -2.0,     // Small negative - CDF ~ 0.023
        0.0,      // Zero - CDF = 0.5
        0.75,     // Small positive (like C6 re-eval) - CDF ~ 0.77
        0.95,     // Small positive (like C5 re-eval) - CDF ~ 0.83
        2.0,      // Moderate positive - CDF ~ 0.977
        10.0,     // Large positive - CDF ~ 1.0
    };

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = testInputs[0];
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    qlforge_test::CumulativeNormalDistributionForge<Real> forgeCDF;
    Real y = forgeCDF(x);
    y.markForgeOutput();
    forge::NodeId yId = y.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    std::cout << "  Graph has " << graph.nodes.size() << " nodes\n";

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    std::cout << "  Kernel compiled, testing...\n\n";

    for (size_t i = 0; i < testInputs.size(); i++) {
        double input = testInputs[i];
        double expected = originalCDF(input);

        double inputVal[4] = {input, input, input, input};
        buffer->setLanes(xId, inputVal);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double outputVal[4];
        buffer->getLanes(yId, outputVal);
        double forgeResult = outputVal[0];

        double relError = std::abs(expected) > 1e-10
            ? std::abs(forgeResult - expected) / std::abs(expected)
            : std::abs(forgeResult - expected);

        bool valueOk = relError < 0.01;
        result.record(valueOk);

        std::cout << "  x=" << std::setw(8) << input
                  << " | Expected: " << std::setw(12) << std::fixed << std::setprecision(8) << expected
                  << " | Forge: " << std::setw(12) << forgeResult
                  << " | Error: " << std::setw(8) << std::scientific << std::setprecision(2) << relError
                  << " | " << (valueOk ? "PASS" : "FAIL")
                  << (i == 0 ? " (build)" : "")
                  << "\n";
    }

    return result;
}

//=============================================================================
// Test extreme negative scenario: build with moderate, re-eval with extreme negative
//=============================================================================
TestResult testExtremeNegativeScenario() {
    TestResult result{"Extreme Negative Scenario"};

    std::cout << "\n=== Testing Extreme Negative Scenario ===\n";
    std::cout << "  Build with moderate x, then re-eval with extreme negative x\n\n";

    std::vector<double> testInputs = {
        1.0,      // Moderate (build)
        0.0,      // Zero
        -1.0,     // Small negative
        -5.0,     // Moderate negative
        -10.0,    // Large negative
        -50.0,    // Very large negative
        -100.0,   // Extreme negative
        -200.0,   // Very extreme negative
    };

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = testInputs[0];
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    qlforge_test::CumulativeNormalDistributionForge<Real> forgeCDF;
    Real y = forgeCDF(x);
    y.markForgeOutput();
    forge::NodeId yId = y.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    std::cout << "  Graph has " << graph.nodes.size() << " nodes\n";

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    std::cout << "  Kernel compiled, testing...\n\n";

    for (size_t i = 0; i < testInputs.size(); i++) {
        double input = testInputs[i];
        double expected = originalCDF(input);

        double inputVal[4] = {input, input, input, input};
        buffer->setLanes(xId, inputVal);
        buffer->clearGradients();
        kernel->execute(*buffer);
        double outputVal[4];
        buffer->getLanes(yId, outputVal);
        double forgeResult = outputVal[0];

        double relError = std::abs(expected) > 1e-10
            ? std::abs(forgeResult - expected) / std::abs(expected)
            : std::abs(forgeResult - expected);

        bool valueOk = relError < 0.01;
        result.record(valueOk);

        std::cout << "  x=" << std::setw(8) << input
                  << " | Expected: " << std::setw(14) << std::scientific << std::setprecision(6) << expected
                  << " | Forge: " << std::setw(14) << forgeResult
                  << " | Error: " << std::setw(8) << std::setprecision(2) << relError
                  << " | " << (valueOk ? "PASS" : "FAIL")
                  << (i == 0 ? " (build)" : "")
                  << "\n";
    }

    return result;
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "=============================================================\n";
    std::cout << "  Minimal CDF Kernel Reuse Test\n";
    std::cout << "=============================================================\n";
    std::cout << "\nApproach:\n";
    std::cout << "  - Original CDF (double): computes expected values\n";
    std::cout << "  - Forge CDF (AReal + ABool::If): builds kernel, re-evaluates\n";
    std::cout << "\nThe Forge CDF uses ABool::If for all branches, so the kernel\n";
    std::cout << "should correctly handle inputs that cross branch boundaries.\n";

    std::vector<TestResult> results;

    results.push_back(testForgeCDFKernelReuse());
    results.push_back(testB4Scenario());
    results.push_back(testReverseScenario());
    results.push_back(testCFormulaScenario());
    results.push_back(testExtremeNegativeScenario());

    std::cout << "\n=============================================================\n";
    std::cout << "  Summary\n";
    std::cout << "=============================================================\n\n";

    int totalTests = 0;
    int passedTests = 0;

    for (const auto& r : results) {
        totalTests += r.total;
        passedTests += r.passed;

        std::cout << "  " << r.name << ": " << r.passed << "/" << r.total << " passed";
        if (!r.allPassed()) {
            std::cout << " *** HAS FAILURES ***";
        }
        std::cout << "\n";
    }

    std::cout << "\n  Total: " << passedTests << "/" << totalTests << " passed\n";
    std::cout << "=============================================================\n";

    if (passedTests == totalTests) {
        std::cout << "\nSUCCESS: Forge CDF kernel reuse works correctly!\n";
        std::cout << "All branch crossings handled properly via ABool::If.\n";
    } else {
        std::cout << "\nFAILURE: Some tests failed.\n";
        std::cout << "Check if ABool::If is properly recording all branches.\n";
    }

    return (passedTests == totalTests) ? 0 : 1;
}
