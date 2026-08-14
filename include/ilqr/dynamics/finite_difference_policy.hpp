#pragma once

#include <utility>

#include "ilqr/math/concepts.hpp"
#include "ilqr/math/finite_differences.hpp"

namespace ilqr
{

/// @brief Differentiation policy (functor): build the discrete Jacobians by central finite
///        differences of the discrete step.
/// @details Perturbs each state (with the control fixed) and each control (with the state fixed),
/// re-evaluating the discrete step for every perturbation, and assembles A = ∂f/∂x and B = ∂f/∂u.
/// Stateless — a compile-time strategy interchangeable with other differentiation policies, mirroring
/// the integration policies. Depends only on the math layer (no third-party libraries), so it is the
/// dependency-light default.
struct FiniteDifference
{
    /// @brief Compute the discrete-step Jacobians A = ∂f_d/∂x, B = ∂f_d/∂u at (x, u).
    /// @details Dimension-agnostic: the state/control vector types are deduced from x and u, and
    /// the Jacobian matrix types follow from them — so no Dims template argument is needed. The
    /// caller packages the returned matrices into a Dims-typed DynamicsTaylorExpansion.
    /// @tparam DiscreteStep_T A discrete-step callable x_{k+1} = f(x, u).
    /// @tparam StateVec_T Deduced state column-vector type.
    /// @tparam ControlVec_T Deduced control column-vector type.
    /// @param f The discrete step to differentiate (already integrates the model over one timestep).
    /// @param x The state to linearize about.
    /// @param u The control to linearize about.
    /// @returns A pair {A, B}: A = ∂step/∂x (state Jacobian), B = ∂step/∂u (control Jacobian).
    template <typename DiscreteStep_T, math::ColumnVector StateVec_T,
              math::ColumnVector ControlVec_T>
        requires math::StateTransition<DiscreteStep_T, StateVec_T, ControlVec_T>
    auto operator()(const DiscreteStep_T& f, const StateVec_T& x, const ControlVec_T& u) const
    {
        // A = ∂f/∂x, holding the control u constant.
        auto A = math::finite_difference_jacobian(
            [&](const StateVec_T& state) { return f(state, u); }, x);
        // B = ∂f/∂u, holding the state x constant.
        auto B = math::finite_difference_jacobian(
            [&](const ControlVec_T& control) { return f(x, control); }, u);
        return std::pair{A, B};
    }
};

}  // namespace ilqr
