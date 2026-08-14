#pragma once

#include <utility>

#include "ilqr/dynamics/dynamics_concepts.hpp"
#include "ilqr/dynamics/finite_difference_policy.hpp"  // for the default DiffPolicy_T = FiniteDifference
#include "ilqr/math/integrators.hpp"

namespace ilqr
{

/// @brief Adapts a continuous-time model into a discrete Dynamics via injected integration (for
/// stepping the discrete model) and differentiation (for computing the dynamic jacobians)
/// strategies/policies.
/// @details Satisfies the Dynamics concept and plugs directly into ILQRSolver. The user writes only
/// a continuous model (a callable ẋ = f(x, u) ) and picks an integration and a
/// differentiation policy. No hand-derived Jacobians are needed. Prefer the discretize() factory
/// over naming this type directly for a cleaner CTAD.
/// @tparam Dims_T Dimensions of the dynamic system.
/// @tparam Model_T A continuous model callable f(x, u) = ẋ (DynamicModel<Model_T, Dims_T>).
/// @tparam IntegrationPolicy_T The integration policy forming the discrete step (default:
///         math::HeunStep — a cheap second-order scheme; use math::RK4Step for more accuracy or
///         math::EulerStep for less evaluations and faster runtime).
/// @tparam DiffPolicy_T The differentiation policy used to build the Jacobians (default:
///         FiniteDifference).
template <typename Dims_T, typename Model_T, typename IntegrationPolicy_T = math::HeunStep,
          typename DiffPolicy_T = FiniteDifference>
    requires DynamicModel<Model_T, Dims_T>
class DiscretizedModel
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims::Scalar;
    using StateVec = typename Dims::StateVec;
    using ControlVec = typename Dims::ControlVec;

    /// @brief Build a discrete model from a continuous one.
    /// @param model The continuous-time model.
    /// @param dt The integration timestep.
    /// @param integrate The integration policy instance (defaults to a default-constructed one).
    /// @param differentiate The differentiation policy instance (defaults to a default-constructed
    ///        one).
    DiscretizedModel(Model_T model, Scalar dt, IntegrationPolicy_T integrate = {},
                     DiffPolicy_T differentiate = {})
        : model_(std::move(model)),
          dt_(dt),
          integrate_(std::move(integrate)),
          differentiate_(std::move(differentiate))
    {
    }

    /// @brief Discrete dynamics x_{k+1} = f_d(x_k, u_k), obtained by integrating the model.
    [[nodiscard]] StateVec step(const StateVec& x, const ControlVec& u) const
    {
        return integrate_(model_, x, u, dt_);
    }

    /// @brief Jacobians A = ∂f_d/∂x, B = ∂f_d/∂u of the discrete step, via the differentiation
    ///        policy.
    [[nodiscard]] DynamicsTaylorExpansion<Dims> linearize(const StateVec& x,
                                                          const ControlVec& u) const
    {
        auto discrete_step = [this](const auto& x_k, const auto& u_k)
        { return integrate_(model_, x_k, u_k, dt_); };

        const auto [A, B] = differentiate_(discrete_step, x, u);
        return {A, B};
    }

private:
    Model_T model_;
    Scalar dt_;
    [[no_unique_address]] IntegrationPolicy_T integrate_;
    [[no_unique_address]] DiffPolicy_T differentiate_;
};

/// @brief Factory to build a discrete Dynamics from a continuous model, a timestep, an integration
/// and a differentiation policy.
/// @details Deduces the model type from the argument; Dims_T must be supplied explicitly because a
/// continuous model (e.g. a lambda) carries no dimension information. The integration and
/// differentiation policies default to math::HeunStep and FiniteDifference and are deduced from
/// their (usually defaulted) tag arguments.
/// @tparam Dims_T Dimensions of the dynamic system.
/// @tparam Model_T Deduced continuous model callable ẋ = f(x, u).
/// @tparam IntegrationPolicy_T Deduced integration policy (default to math::HeunStep).
/// @tparam DiffPolicy_T Deduced differentiation policy (default to FiniteDifference).
/// @param model The continuous-time model
/// @param dt The integration timestep
/// @param integrate The integration policy instance
/// @param differentiate The differentiation policy instance
/// @returns A DiscretizedModel satisfying the Dynamics concept.
template <typename Dims_T, typename Model_T, typename IntegrationPolicy_T = math::HeunStep,
          typename DiffPolicy_T = FiniteDifference>
    requires DynamicModel<Model_T, Dims_T>
[[nodiscard]] auto discretize(Model_T model, typename Dims_T::Scalar dt,
                              IntegrationPolicy_T integrate = {}, DiffPolicy_T differentiate = {})
    -> DiscretizedModel<Dims_T, Model_T, IntegrationPolicy_T, DiffPolicy_T>
{
    return {std::move(model), dt, std::move(integrate), std::move(differentiate)};
}

}  // namespace ilqr
