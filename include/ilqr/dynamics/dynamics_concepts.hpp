#pragma once

#include <concepts>
#include <Eigen/Dense>

#include "ilqr/core/types.hpp"
#include "ilqr/math/concepts.hpp"

namespace ilqr
{

/// @brief First-order Taylor expansion of the system dynamics given by
/// f_d(x̄+δx, ū+δu) ≈ f(x̄,ū) + A·δx + B·δu
/// @tparam Dims_T Dimensions of dynamic system
template <typename Dims_T>
struct DynamicsTaylorExpansion
{
    /// State jacobian ∂f/∂x
    typename Dims_T::StateMat A{Dims_T::StateMat::Zero()};
    /// Control jacobian ∂f/∂u
    typename Dims_T::StateControlMat B{Dims_T::StateControlMat::Zero()};
};

/// @brief Dynamics contract requires to publish its dimensions and
/// - A step(x, u) -> x computing the next state from the discrete system dynamics via x_k+1 =
/// f_d(x_k, u_k)
/// - A linearize(x, u) -> DynamicsTaylorExpansion returning the first-order taylor expansion of the
/// discrete dynamics f_d(x_k, u_k)
template <typename Dynamics_T>
concept Dynamics = requires { typename Dynamics_T::Dims; } &&
                   requires(const Dynamics_T dyn, const typename Dynamics_T::Dims::StateVec& x,
                            const typename Dynamics_T::Dims::ControlVec& u) {
                       {
                           dyn.step(x, u)
                       } -> std::convertible_to<typename Dynamics_T::Dims::StateVec>;
                       {
                           dyn.linearize(x, u)
                       } -> std::convertible_to<DynamicsTaylorExpansion<typename Dynamics_T::Dims>>;
                   };

/// @brief A dynamic model: a callable (state, control) -> state, usable as either a continuous
/// derivative ẋ = f(x, u) or a discrete step x_{k+1} = f(x, u).
/// @details Refinement of math::StateTransition expressed in terms of the problem dimensions: it
/// requires f to accept Dims_T::StateVec and Dims_T::ControlVec and return a Dims_T::StateVec.
/// @tparam Fn A callable invoked as f(x, u).
/// @tparam Dims_T Dimensions of the dynamic system.
template <typename Fn, typename Dims_T>
concept DynamicModel =
    math::StateTransition<Fn, typename Dims_T::StateVec, typename Dims_T::ControlVec>;

}  // namespace ilqr
