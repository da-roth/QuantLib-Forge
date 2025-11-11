// This file is part of QuantLib-Forge <https://github.com/da-roth/QuantLib-Forge>
//
// Adapter to integrate Forge AAD engine with QuantLib
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file session.hpp
 * @brief RAII wrapper for Forge recording/compilation/execution workflow
 *
 * Provides a simple session interface to:
 * - Start/stop graph recording
 * - Mark inputs and outputs
 * - Compile the recorded graph
 * - Execute and extract values/adjoints
 */

#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "../../../forge/tools/types/fdouble.hpp"
#include "../../../forge/src/graph/graph.hpp"
#include "../../../forge/src/graph/graph_recorder.hpp"
#include "../../../forge/src/graph/handles.hpp"
#include "../../../forge/src/compiler/forge_engine.hpp"
#include "../../../forge/src/compiler/node_value_buffers/node_value_buffer.hpp"
#include "../../../forge/src/compiler/compiler_config.hpp"

namespace qlforge {

/**
 * @brief Session manager for Forge AAD computations
 *
 * Usage pattern:
 * 1. Create session (starts recording)
 * 2. Mark independent variables with markInput()
 * 3. Execute QuantLib pricing code
 * 4. Mark outputs with markOutput()
 * 5. Call finalize() to compile
 * 6. Call execute() to run with specific input values
 * 7. Extract results with getOutputs() and getAdjoints()
 *
 * Example:
 * @code
 * Session sess;
 * forge::fdouble S(100.0), r(0.01), vol(0.20);
 *
 * // Mark inputs
 * auto h_S = sess.markInput(S);
 * auto h_r = sess.markInput(r);
 * auto h_vol = sess.markInput(vol);
 *
 * // Price option (QuantLib code using S, r, vol)
 * forge::fdouble price = computeOptionPrice(S, r, vol);
 *
 * // Mark output
 * auto h_price = sess.markOutput(price);
 *
 * // Compile
 * sess.finalize();
 *
 * // Execute with values
 * sess.setInputValue(h_S, 100.0);
 * sess.setInputValue(h_r, 0.01);
 * sess.setInputValue(h_vol, 0.20);
 * sess.execute();
 *
 * // Extract results
 * double npv = sess.getOutputValue(h_price);
 * double delta = sess.getAdjoint(h_S);
 * @endcode
 */
class Session {
private:
    forge::GraphRecorder recorder_;
    std::unique_ptr<forge::StitchedKernel> kernel_;
    std::unique_ptr<forge::INodeValueBuffer> buffer_;
    bool finalized_ = false;

    // Handle storage
    std::vector<forge::InputHandle> inputHandles_;
    std::vector<forge::ResultHandle> outputHandles_;

public:
    /**
     * @brief Constructor - starts recording
     */
    Session() {
        recorder_.start();
    }

    /**
     * @brief Destructor - stops recording if not finalized
     */
    ~Session() {
        if (!finalized_ && recorder_.isRecording()) {
            recorder_.stop();
        }
    }

    // Prevent copying
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    /**
     * @brief Mark a variable as an independent input
     * @param x The fdouble to mark as input
     * @return InputHandle for later reference
     */
    forge::InputHandle markInput(forge::fdouble& x) {
        if (!recorder_.isRecording()) {
            throw std::runtime_error("Cannot mark input: recording not active");
        }
        auto handle = x.markInputAndDiff();  // Mark for gradient computation
        inputHandles_.push_back(handle);
        return handle;
    }

    /**
     * @brief Mark a variable as an output
     * @param x The fdouble to mark as output
     * @return ResultHandle for later reference
     */
    forge::ResultHandle markOutput(const forge::fdouble& x) {
        if (!recorder_.isRecording()) {
            throw std::runtime_error("Cannot mark output: recording not active");
        }
        auto handle = x.markOutput();
        outputHandles_.push_back(handle);
        return handle;
    }

    /**
     * @brief Finalize recording and compile the graph
     *
     * This stops recording, optimizes the graph, and compiles
     * it to executable machine code.
     */
    void finalize() {
        if (finalized_) {
            throw std::runtime_error("Session already finalized");
        }
        if (!recorder_.isRecording()) {
            throw std::runtime_error("Cannot finalize: not recording");
        }

        // Stop recording
        recorder_.stop();

        // Get the recorded graph
        const forge::Graph& graph = recorder_.graph();

        // Compile with default config (can be customized)
        forge::CompilerConfig config = forge::CompilerConfig::Default();
        forge::ForgeEngine engine(config);

        std::cout << "[QLForge Session] Compiling graph with "
                  << inputHandles_.size() << " inputs and "
                  << outputHandles_.size() << " outputs..." << std::endl;

        kernel_ = engine.compile(graph);

        // Create value buffer
        buffer_ = forge::NodeValueBufferFactory::create(graph, *kernel_);

        finalized_ = true;

        std::cout << "[QLForge Session] Compilation complete. Buffer size: "
                  << buffer_->getNumNodes() << " nodes" << std::endl;
    }

    /**
     * @brief Set value for an input
     * @param handle The InputHandle returned from markInput()
     * @param value The value to set
     */
    void setInputValue(const forge::InputHandle& handle, double value) {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized. Call finalize() first.");
        }
        buffer_->setValue(handle.nodeId, value);
    }

    /**
     * @brief Set all input values at once
     * @param values Vector of values in the same order as markInput() calls
     */
    void setInputValues(const std::vector<double>& values) {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized. Call finalize() first.");
        }
        if (values.size() != inputHandles_.size()) {
            throw std::runtime_error("Number of values doesn't match number of inputs");
        }
        for (size_t i = 0; i < values.size(); ++i) {
            buffer_->setValue(inputHandles_[i].nodeId, values[i]);
        }
    }

    /**
     * @brief Execute the compiled kernel
     *
     * Runs both forward pass (computes outputs) and backward pass
     * (computes gradients).
     */
    void execute() {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized. Call finalize() first.");
        }
        if (!kernel_) {
            throw std::runtime_error("No compiled kernel available");
        }

        // Clear previous gradients
        buffer_->clearGradients();

        // Execute kernel
        kernel_->execute(*buffer_);
    }

    /**
     * @brief Get output value
     * @param handle The ResultHandle returned from markOutput()
     * @return The computed output value
     */
    double getOutputValue(const forge::ResultHandle& handle) const {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized");
        }
        return buffer_->getValue(handle.nodeId);
    }

    /**
     * @brief Get all output values
     * @return Vector of output values in order of markOutput() calls
     */
    std::vector<double> getOutputValues() const {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized");
        }
        std::vector<double> results;
        results.reserve(outputHandles_.size());
        for (const auto& handle : outputHandles_) {
            results.push_back(buffer_->getValue(handle.nodeId));
        }
        return results;
    }

    /**
     * @brief Get adjoint (gradient) for an input
     * @param handle The InputHandle returned from markInput()
     * @return The gradient ∂output/∂input
     */
    double getAdjoint(const forge::InputHandle& handle) const {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized");
        }
        if (!buffer_->hasGradients()) {
            throw std::runtime_error("No gradients available. Did you call execute()?");
        }
        return buffer_->getGradient(handle.nodeId);
    }

    /**
     * @brief Get all adjoints
     * @return Vector of gradients in order of markInput() calls
     */
    std::vector<double> getAdjoints() const {
        if (!finalized_) {
            throw std::runtime_error("Session not finalized");
        }
        if (!buffer_->hasGradients()) {
            throw std::runtime_error("No gradients available. Did you call execute()?");
        }
        std::vector<double> grads;
        grads.reserve(inputHandles_.size());
        for (const auto& handle : inputHandles_) {
            grads.push_back(buffer_->getGradient(handle.nodeId));
        }
        return grads;
    }

    /**
     * @brief Get the number of inputs
     */
    size_t numInputs() const { return inputHandles_.size(); }

    /**
     * @brief Get the number of outputs
     */
    size_t numOutputs() const { return outputHandles_.size(); }

    /**
     * @brief Check if session is finalized
     */
    bool isFinalized() const { return finalized_; }

    /**
     * @brief Check if currently recording
     */
    bool isRecording() const { return recorder_.isRecording(); }
};

} // namespace qlforge
