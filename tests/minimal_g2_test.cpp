/*******************************************************************************

   Minimal G2 Re-evaluation Test

   This file tests the core issue: gradients failing on re-evaluation with
   different inputs. It uses only AReal + Forge, no QuantLib dependency.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#include <expressions/Literals.hpp>
#include <expressions/ExpressionTemplates/BinaryOperators.hpp>
#include <expressions/ExpressionTemplates/UnaryOperators.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using Real = forge::expr::AReal<double>;

// Test result tracking
struct TestResult {
    std::string name;
    bool v1_ok, g1_ok, v2_ok, g2_ok;

    bool passed() const { return v1_ok && g1_ok && v2_ok && g2_ok; }

    void print() const {
        std::cout << name << ": "
                  << "V1=" << (v1_ok ? "OK" : "FAIL") << " "
                  << "G1=" << (g1_ok ? "OK" : "FAIL") << " "
                  << "V2=" << (v2_ok ? "OK" : "FAIL") << " "
                  << "G2=" << (g2_ok ? "OK" : "FAIL")
                  << (passed() ? "" : " *** FAILED ***")
                  << "\n";
    }
};

//=============================================================================
// Test 1: Basic Arithmetic (z = x * y + x)
// dz/dx = y + 1, dz/dy = x
// This gradient DEPENDS on input values - should fail G2 if bug exists
//=============================================================================
TestResult testBasicArithmetic() {
    TestResult result{"BasicArithmetic", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 2.0, y = 3.0;
    x.markForgeInputAndDiff();
    y.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId(), yId = y.forgeNodeId();

    Real z = x * y + x;
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId), buffer->getBufferIndex(yId)};
    std::vector<double> grad(2 * vw);

    // Test 1: x=2, y=3 -> z=8, dz/dx=4, dz/dy=2
    double xVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal1);
    double yVal1[4] = {3.0, 3.0, 3.0, 3.0}; buffer->setLanes(yId, yVal1);
    buffer->clearGradients();  // Must clear before execute!
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    double z1 = zOut1[0];
    result.v1_ok = std::abs(z1 - 8.0) < 1e-9;
    result.g1_ok = std::abs(grad[0] - 4.0) < 1e-9 && std::abs(grad[1 * vw] - 2.0) < 1e-9;

    // Test 2 (re-eval): x=4, y=5 -> z=24, dz/dx=6, dz/dy=4
    double xVal2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal2);
    double yVal2[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(yId, yVal2);
    buffer->clearGradients();  // Must clear before execute!
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    double z2 = zOut2[0];
    result.v2_ok = std::abs(z2 - 24.0) < 1e-9;
    result.g2_ok = std::abs(grad[0] - 6.0) < 1e-9 && std::abs(grad[1 * vw] - 4.0) < 1e-9;

    if (!result.g2_ok) {
        std::cout << "  [DEBUG] Expected dz/dx=6, dz/dy=4, got dz/dx=" << grad[0] << ", dz/dy=" << grad[1] << "\n";
    }

    return result;
}

//=============================================================================
// Test 2: Simple Pass-through (z = x)
// dz/dx = 1 (constant)
// This gradient does NOT depend on input - should pass G2
//=============================================================================
TestResult testPassThrough() {
    TestResult result{"PassThrough", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 100.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real z = x;  // Just pass through
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=100 -> z=100, dz/dx=1
    double xVal1[4] = {100.0, 100.0, 100.0, 100.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    result.v1_ok = std::abs(zOut1[0] - 100.0) < 1e-9;
    result.g1_ok = std::abs(grad[0] - 1.0) < 1e-9;

    // Test 2: x=200 -> z=200, dz/dx=1
    double xVal2[4] = {200.0, 200.0, 200.0, 200.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    result.v2_ok = std::abs(zOut2[0] - 200.0) < 1e-9;
    result.g2_ok = std::abs(grad[0] - 1.0) < 1e-9;

    return result;
}

//=============================================================================
// Test 3: Log function (z = log(x))
// dz/dx = 1/x (depends on x)
// This gradient DEPENDS on input - should fail G2 if bug exists
//=============================================================================
TestResult testLog() {
    TestResult result{"Log", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 2.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real z = forge::expr::log(x);
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=2 -> z=log(2), dz/dx=0.5
    double xVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    result.v1_ok = std::abs(zOut1[0] - std::log(2.0)) < 1e-9;
    result.g1_ok = std::abs(grad[0] - 0.5) < 1e-9;

    // Test 2: x=4 -> z=log(4), dz/dx=0.25
    double xVal2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    result.v2_ok = std::abs(zOut2[0] - std::log(4.0)) < 1e-9;
    result.g2_ok = std::abs(grad[0] - 0.25) < 1e-9;

    if (!result.g2_ok) {
        std::cout << "  [DEBUG] Expected dz/dx=0.25, got dz/dx=" << grad[0] << "\n";
    }

    return result;
}

//=============================================================================
// Test 4: Exp function (z = exp(x))
// dz/dx = exp(x) (depends on x)
// This gradient DEPENDS on input - should fail G2 if bug exists
//=============================================================================
TestResult testExp() {
    TestResult result{"Exp", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 1.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real z = forge::expr::exp(x);
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=1 -> z=e, dz/dx=e
    double xVal1[4] = {1.0, 1.0, 1.0, 1.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    result.v1_ok = std::abs(zOut1[0] - std::exp(1.0)) < 1e-9;
    result.g1_ok = std::abs(grad[0] - std::exp(1.0)) < 1e-9;

    // Test 2: x=2 -> z=e^2, dz/dx=e^2
    double xVal2[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    result.v2_ok = std::abs(zOut2[0] - std::exp(2.0)) < 1e-9;
    result.g2_ok = std::abs(grad[0] - std::exp(2.0)) < 1e-9;

    if (!result.g2_ok) {
        std::cout << "  [DEBUG] Expected dz/dx=" << std::exp(2.0) << ", got dz/dx=" << grad[0] << "\n";
    }

    return result;
}

//=============================================================================
// Test 5: Division (z = 1/x)
// dz/dx = -1/x^2 (depends on x)
//=============================================================================
TestResult testDivision() {
    TestResult result{"Division", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 2.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real z = Real(1.0) / x;
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=2 -> z=0.5, dz/dx=-0.25
    double xVal1[4] = {2.0, 2.0, 2.0, 2.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    result.v1_ok = std::abs(zOut1[0] - 0.5) < 1e-9;
    result.g1_ok = std::abs(grad[0] - (-0.25)) < 1e-9;

    // Test 2: x=4 -> z=0.25, dz/dx=-0.0625
    double xVal2[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    result.v2_ok = std::abs(zOut2[0] - 0.25) < 1e-9;
    result.g2_ok = std::abs(grad[0] - (-0.0625)) < 1e-9;

    if (!result.g2_ok) {
        std::cout << "  [DEBUG] Expected dz/dx=-0.0625, got dz/dx=" << grad[0] << "\n";
    }

    return result;
}

//=============================================================================
// Test 6: Sqrt (z = sqrt(x))
// dz/dx = 0.5/sqrt(x) (depends on x)
//=============================================================================
TestResult testSqrt() {
    TestResult result{"Sqrt", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 4.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real z = forge::expr::sqrt(x);
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=4 -> z=2, dz/dx=0.25
    double xVal1[4] = {4.0, 4.0, 4.0, 4.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    result.v1_ok = std::abs(zOut1[0] - 2.0) < 1e-9;
    result.g1_ok = std::abs(grad[0] - 0.25) < 1e-9;

    // Test 2: x=16 -> z=4, dz/dx=0.125
    double xVal2[4] = {16.0, 16.0, 16.0, 16.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    result.v2_ok = std::abs(zOut2[0] - 4.0) < 1e-9;
    result.g2_ok = std::abs(grad[0] - 0.125) < 1e-9;

    if (!result.g2_ok) {
        std::cout << "  [DEBUG] Expected dz/dx=0.125, got dz/dx=" << grad[0] << "\n";
    }

    return result;
}

//=============================================================================
// Test 7: Linear (z = 2*x + 3)
// dz/dx = 2 (constant)
// Should PASS G2 since gradient is constant
//=============================================================================
TestResult testLinear() {
    TestResult result{"Linear", false, false, false, false};

    forge::GraphRecorder recorder;
    recorder.start();

    Real x = 5.0;
    x.markForgeInputAndDiff();
    forge::NodeId xId = x.forgeNodeId();

    Real z = Real(2.0) * x + Real(3.0);
    z.markForgeOutput();
    forge::NodeId zId = z.forgeNodeId();

    recorder.stop();
    forge::Graph graph = recorder.graph();

    forge::ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

    int vw = buffer->getVectorWidth();
    std::vector<size_t> gradIdx = {buffer->getBufferIndex(xId)};
    std::vector<double> grad(1 * vw);

    // Test 1: x=5 -> z=13, dz/dx=2
    double xVal1[4] = {5.0, 5.0, 5.0, 5.0}; buffer->setLanes(xId, xVal1);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut1[4]; buffer->getLanes(zId, zOut1);
    result.v1_ok = std::abs(zOut1[0] - 13.0) < 1e-9;
    result.g1_ok = std::abs(grad[0] - 2.0) < 1e-9;

    // Test 2: x=10 -> z=23, dz/dx=2
    double xVal2[4] = {10.0, 10.0, 10.0, 10.0}; buffer->setLanes(xId, xVal2);
    buffer->clearGradients();
    kernel->execute(*buffer);
    buffer->getGradientLanes(gradIdx, grad.data());

    double zOut2[4]; buffer->getLanes(zId, zOut2);
    result.v2_ok = std::abs(zOut2[0] - 23.0) < 1e-9;
    result.g2_ok = std::abs(grad[0] - 2.0) < 1e-9;

    return result;
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "=============================================================\n";
    std::cout << "  Minimal G2 Re-evaluation Test (AReal + Forge only)\n";
    std::cout << "=============================================================\n\n";

    std::vector<TestResult> results;

    results.push_back(testBasicArithmetic());
    results.push_back(testPassThrough());
    results.push_back(testLog());
    results.push_back(testExp());
    results.push_back(testDivision());
    results.push_back(testSqrt());
    results.push_back(testLinear());

    std::cout << "\nResults:\n";
    std::cout << "-------------------------------------------------------------\n";

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        r.print();
        if (r.passed()) ++passed; else ++failed;
    }

    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Passed: " << passed << "/" << results.size() << "\n";
    std::cout << "Failed: " << failed << "/" << results.size() << "\n";
    std::cout << "=============================================================\n";

    // Expected pattern:
    // - Constant gradient tests (PassThrough, Linear) should PASS
    // - Input-dependent gradient tests (BasicArithmetic, Log, Exp, Division, Sqrt) should FAIL G2

    std::cout << "\nExpected behavior if G2 bug exists:\n";
    std::cout << "  PassThrough, Linear -> PASS (constant gradients)\n";
    std::cout << "  BasicArithmetic, Log, Exp, Division, Sqrt -> FAIL G2 (input-dependent gradients)\n";

    return failed > 0 ? 1 : 0;
}
