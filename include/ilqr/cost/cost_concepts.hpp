#pragma once

#include <concepts>
#include <Eigen/Dense>

#include "ilqr/core/types.hpp"

namespace ilqr
{
/// @brief Second-order Taylor expansion of the running cost l_k(x,u) at a given state-control
/// pair (x, u) at a timestep k
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
struct CostTaylorExpansion
{
    /// Aliases
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using StateMat = typename Dims_T::StateMat;
    using ControlMat = typename Dims_T::ControlMat;
    using ControlStateMat = typename Dims_T::ControlStateMat;

    /// Cost value
    Scalar l{Scalar(0)};
    /// State cost gradient ∂l/∂x
    StateVec l_x{StateVec::Zero()};
    /// Control cost gradient ∂l/∂u
    ControlVec l_u{ControlVec::Zero()};
    /// State cost hessian ∂²l/∂x²
    StateMat l_xx{StateMat::Zero()};
    /// Control cost hessian ∂²l/∂u²
    ControlMat l_uu{ControlMat::Zero()};
    /// Control state cost hessian ∂²l/∂u∂x
    ControlStateMat l_ux{ControlStateMat::Zero()};

    /// @brief Addition assignment operator overload
    /// @param rhs The cost expansions to be added
    /// @returns A reference to the modified cost expansion
    CostTaylorExpansion& operator+=(const CostTaylorExpansion& rhs);
};

/// @brief Second-order Taylor expansion of the final cost l_f(x_N) at a given final state x_N
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
struct FinalCostTaylorExpansion
{
    /// Aliases
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using StateMat = typename Dims_T::StateMat;

    /// Final cost value
    Scalar lf{Scalar(0)};
    /// State final cost gradient ∂lf/∂x
    StateVec lf_x{StateVec::Zero()};
    /// State final cost hessian ∂²lf/∂x²
    StateMat lf_xx{StateMat::Zero()};

    /// @brief Addition assignment operator overload
    /// @param rhs The final cost expansions to be added
    /// @returns A reference to the modified final cost expansion
    FinalCostTaylorExpansion& operator+=(const FinalCostTaylorExpansion& rhs);
};

/// @brief Concept contract for a cost function
/// @tparam Cost_T The cost function candidate
/// @details A cost function needs to expose its dimensions as well as four methods
/// - An evaluate(x, u, k) -> Scalar method that evaluates the cost at a given state, control and
/// timestep
/// - A quadratize(x, u, k) -> CostTaylorExpansion method that returns the second-order Taylor
/// approximation terms of the cost function at a given state, control and timestep
/// - An evaluate_final(x) -> Scalar method that evaluates the final cost at the final state
/// - A quadratize_final(x) -> FinalCostTaylorExpansion method that returns the second-order
/// Taylor approximation of the final cost at a given final state
template <typename Cost_T>
concept CostFunction =
    requires { typename Cost_T::Dims; } &&
    requires(const Cost_T cost, const typename Cost_T::Dims::StateVec& x,
             const typename Cost_T::Dims::ControlVec& u, int k) {
        {
            cost.evaluate(x, u, k)
        } -> std::convertible_to<typename Cost_T::Dims::Scalar>;
        {
            cost.quadratize(x, u, k)
        } -> std::convertible_to<CostTaylorExpansion<typename Cost_T::Dims>>;
        {
            cost.evaluate_final(x)
        } -> std::convertible_to<typename Cost_T::Dims::Scalar>;
        {
            cost.quadratize_final(x)
        } -> std::convertible_to<FinalCostTaylorExpansion<typename Cost_T::Dims>>;
    };

}  // namespace ilqr

#include "ilqr/cost/cost_concepts.inl"
