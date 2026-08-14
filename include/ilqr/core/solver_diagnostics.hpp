#pragma once

#include <concepts>
#include <limits>
#include <vector>

namespace ilqr
{

/// @brief Snapshot of a single iLQR iteration
template <std::floating_point Scalar_T>
struct IterationRecord
{
    // Trajectory cost (entering cost if not accepted, post-step cost if accepted)
    Scalar_T cost{std::numeric_limits<Scalar_T>::quiet_NaN()};
    // Regularization in effect for this iteration's backward pass
    Scalar_T reg{std::numeric_limits<Scalar_T>::quiet_NaN()};
    // Normalized gradient (feedforward) norm; NaN if the backward pass failed before it
    Scalar_T grad_norm{std::numeric_limits<Scalar_T>::quiet_NaN()};
    // Accepted line search step size; NaN unless a step was accepted
    Scalar_T step_size{std::numeric_limits<Scalar_T>::quiet_NaN()};
    // Whether the forward pass accepted a step this iteration
    bool accepted_forward_pass{false};
};

/// @brief Per-solve diagnostics: one record per outer iteration (including iterations that
/// exit early without an accepted step). NaN fields mark quantities not computed that iteration.
template <std::floating_point Scalar_T>
struct SolveDiagnostics
{
    // History of iterations; iterations.size() is the number of iterations run
    std::vector<IterationRecord<Scalar_T>> iterations;
};

namespace details
{

/// @brief RAII guard that appends its record to the diagnostics (when present) on scope exit,
/// allowing a non-intrusive per-iteration recording of a solver iteration. Implementation detail
/// of the solver; not part of the public diagnostics API.
template <std::floating_point Scalar_T>
struct IterationRecorder
{
    // Non-owning pointer to diagnostics
    SolveDiagnostics<Scalar_T>* diagnostics;
    // The payload to be recorded in one iteration
    IterationRecord<Scalar_T> record{};

    /// @brief Construct an iteration recorder
    explicit IterationRecorder(SolveDiagnostics<Scalar_T>* diagnostics_) : diagnostics(diagnostics_)
    {
    }

    /// @brief Non-copyable guard - avoids multiple push_back actions
    IterationRecorder(const IterationRecorder&) = delete;
    IterationRecorder& operator=(const IterationRecorder&) = delete;

    /// @brief Destructor flushes an iteration record into the diagnostics (in case one exists)
    ~IterationRecorder()
    {
        if (diagnostics) diagnostics->iterations.push_back(record);
    }
};

}  // namespace details

}  // namespace ilqr
