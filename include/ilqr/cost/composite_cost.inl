#pragma once

#include "ilqr/cost/composite_cost.hpp"

namespace ilqr
{

template <CostFunction... CostTerms_T>
CompositeCostFunction<CostTerms_T...>::CompositeCostFunction(CostTerms_T... terms)
    : terms_(std::move(terms)...)
{
}

template <CostFunction... CostTerms_T>
auto CompositeCostFunction<CostTerms_T...>::evaluate(const StateVec& state,
                                                     const ControlVec& control, int k) const
    -> Scalar
{
    Scalar sum = Scalar(0);
    std::apply([&](const CostTerms_T&... term)
               { ((sum += term.evaluate(state, control, k)), ...); },
               terms_);
    return sum;
}

template <CostFunction... CostTerms_T>
auto CompositeCostFunction<CostTerms_T...>::quadratize(const StateVec& state,
                                                       const ControlVec& control, int k) const
    -> CostTaylorExpansion<Dims>
{
    auto accumulated_cost_expansion = CostTaylorExpansion<Dims>{};
    std::apply([&](const CostTerms_T&... term)
               { ((accumulated_cost_expansion += term.quadratize(state, control, k)), ...); },
               terms_);
    return accumulated_cost_expansion;
}

template <CostFunction... CostTerms_T>
auto CompositeCostFunction<CostTerms_T...>::evaluate_final(const StateVec& state) const -> Scalar
{
    Scalar sum = Scalar(0);
    std::apply([&](const CostTerms_T&... term) { ((sum += term.evaluate_final(state)), ...); },
               terms_);
    return sum;
}

template <CostFunction... CostTerms_T>
auto CompositeCostFunction<CostTerms_T...>::quadratize_final(const StateVec& state) const
    -> FinalCostTaylorExpansion<Dims>
{
    auto accumulated_final_cost_expansion = FinalCostTaylorExpansion<Dims>{};
    std::apply([&](const CostTerms_T&... term)
               { ((accumulated_final_cost_expansion += term.quadratize_final(state)), ...); },
               terms_);
    return accumulated_final_cost_expansion;
}

}  // namespace ilqr
