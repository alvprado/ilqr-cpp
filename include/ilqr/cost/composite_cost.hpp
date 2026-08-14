#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "ilqr/cost/cost_concepts.hpp"

namespace ilqr
{

/// @brief Composes a single cost function from multiple cost terms; the total cost is
/// the sum of the terms' contributions.
/// @tparam CostTerms_T Pack of cost-term types; each must satisfy the CostFunction
/// concept, and all must share the same Dims.
template <CostFunction... CostTerms_T>
class CompositeCostFunction
{
public:
    /// Check there is at least one cost term
    static_assert(sizeof...(CostTerms_T) > 0, "CompositeCostFunction needs at least one term.");

    /// Adopt the dimensions of the first cost term - all terms must agree
    using Dims = typename std::tuple_element_t<0, std::tuple<CostTerms_T...>>::Dims;

    /// Check all cost terms share the same dimensions
    static_assert(((std::is_same_v<Dims, typename CostTerms_T::Dims>)&&...),
                  "All cost terms must have the same dimensions.");

    /// Aliases
    using Scalar = typename Dims::Scalar;
    using StateVec = typename Dims::StateVec;
    using ControlVec = typename Dims::ControlVec;

    /// @brief Ctor for initializing a CompositeCostFunction with a parameter pack of cost terms
    /// @param terms parameter pack of cost terms
    explicit CompositeCostFunction(CostTerms_T... terms);

    /// @brief Delet default Ctor to avoid empty cost terms
    CompositeCostFunction() = delete;

    /// @brief Evaluates the running cost at a state-control pair, summed over all terms.
    /// @param state The state x at timestep k
    /// @param control The control u at timestep k
    /// @param k The timestep index
    /// @return The combined running cost l_k(x, u)
    [[nodiscard]] Scalar evaluate(const StateVec& state, const ControlVec& control, int k) const;

    /// @brief Builds the second-order Taylor expansion of the running cost at (x, u, k),
    /// summed over all terms.
    /// @param state The state x at timestep k
    /// @param control The control u at timestep k
    /// @param k The timestep index
    /// @return The combined running-cost expansion (value, gradients and Hessians)
    [[nodiscard]] CostTaylorExpansion<Dims> quadratize(const StateVec& state,
                                                       const ControlVec& control, int k) const;

    /// @brief Evaluates the final cost at the terminal state, summed over all terms.
    /// @param state The final state x_N
    /// @return The combined final cost l_f(x_N)
    [[nodiscard]] Scalar evaluate_final(const StateVec& state) const;

    /// @brief Builds the second-order Taylor expansion of the final cost at x_N,
    /// summed over all terms.
    /// @param state The final state x_N
    /// @return The combined final-cost expansion (value, gradient and Hessian)
    [[nodiscard]] FinalCostTaylorExpansion<Dims> quadratize_final(const StateVec& state) const;

private:
    std::tuple<CostTerms_T...> terms_;
};

}  // namespace ilqr

#include "ilqr/cost/composite_cost.inl"
