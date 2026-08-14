#pragma once

#include "ilqr/math/concepts.hpp"

namespace ilqr::math
{

/// @brief Integration policy: a single first-order explicit (forward) Euler step.
struct EulerStep
{
    /// @brief Advance the state by one timestep with forward Euler.
    /// @tparam Model_T A callable ẋ = f(x, u).
    /// @tparam StateVec_T The state column-vector type.
    /// @tparam ControlVec_T The control column-vector type.
    /// @param[in] f The continuous model ẋ = f(x, u).
    /// @param[in] x The state at the start of the step.
    /// @param[in] u The control held constant over the step.
    /// @param[in] dt The timestep.
    /// @returns The state advanced by one timestep.
    template <typename Model_T, ColumnVector StateVec_T, ColumnVector ControlVec_T>
        requires StateTransition<Model_T, StateVec_T, ControlVec_T>
    StateVec_T operator()(const Model_T& f, const StateVec_T& x, const ControlVec_T& u,
                          typename StateVec_T::Scalar dt) const
    {
        const StateVec_T k1 = f(x, u);
        return x + dt * k1;
    }
};

/// @brief Integration policy: a single second-order explicit-trapezoidal (Heun) step.
struct HeunStep
{
    /// @brief Advance the state by one timestep with Heun's method (two stages).
    /// @tparam Model_T A callable ẋ = f(x, u).
    /// @tparam StateVec_T The state column-vector type.
    /// @tparam ControlVec_T The control column-vector type.
    /// @param[in] f The continuous model ẋ = f(x, u).
    /// @param[in] x The state at the start of the step.
    /// @param[in] u The control held constant over the step.
    /// @param[in] dt The timestep.
    /// @returns The state advanced by one timestep.
    template <typename Model_T, ColumnVector StateVec_T, ColumnVector ControlVec_T>
        requires StateTransition<Model_T, StateVec_T, ControlVec_T>
    StateVec_T operator()(const Model_T& f, const StateVec_T& x, const ControlVec_T& u,
                          typename StateVec_T::Scalar dt) const
    {
        using Scalar = typename StateVec_T::Scalar;

        // Evaluate each stage state into a concrete vector before calling the model, so the model
        // always receives a plain vector rather than an unevaluated Eigen expression.
        const StateVec_T k1 = f(x, u);
        const StateVec_T x2 = x + dt * k1;
        const StateVec_T k2 = f(x2, u);
        return x + (dt / Scalar(2)) * (k1 + k2);
    }
};

/// @brief Integration policy: a single fourth-order Runge-Kutta (RK4) step.
struct RK4Step
{
    /// @brief Advance the state by one timestep with classic RK4 (four stages).
    /// @tparam Model_T A callable ẋ = f(x, u) .
    /// @tparam StateVec_T The state column-vector type.
    /// @tparam ControlVec_T The control column-vector type.
    /// @param[in] f The continuous model ẋ = f(x, u).
    /// @param[in] x The state at the start of the step.
    /// @param[in] u The control held constant over the step.
    /// @param[in] dt The timestep.
    /// @returns The state advanced by one timestep.
    template <typename Model_T, ColumnVector StateVec_T, ColumnVector ControlVec_T>
        requires StateTransition<Model_T, StateVec_T, ControlVec_T>
    StateVec_T operator()(const Model_T& f, const StateVec_T& x, const ControlVec_T& u,
                          typename StateVec_T::Scalar dt) const
    {
        using Scalar = typename StateVec_T::Scalar;

        // Evaluate each stage state into a concrete vector before calling the model, so the model
        // always receives a plain vector rather than an unevaluated Eigen expression.
        const StateVec_T k1 = f(x, u);
        const StateVec_T x2 = x + (dt / Scalar(2)) * k1;
        const StateVec_T k2 = f(x2, u);
        const StateVec_T x3 = x + (dt / Scalar(2)) * k2;
        const StateVec_T k3 = f(x3, u);
        const StateVec_T x4 = x + dt * k3;
        const StateVec_T k4 = f(x4, u);
        return x + (dt / Scalar(6)) * (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4);
    }
};

}  // namespace ilqr::math
