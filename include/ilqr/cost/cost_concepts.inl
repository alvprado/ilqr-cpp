#pragma once

#include "ilqr/cost/cost_concepts.hpp"

namespace ilqr
{

// ─── Cost Taylor Expansion ──────────────────────────────────────────────────────

template <typename Dims_T>
auto CostTaylorExpansion<Dims_T>::operator+=(const CostTaylorExpansion& rhs) -> CostTaylorExpansion&
{
    l += rhs.l;
    l_x += rhs.l_x;
    l_u += rhs.l_u;
    l_xx += rhs.l_xx;
    l_uu += rhs.l_uu;
    l_ux += rhs.l_ux;
    return *this;
}

// ─── Final Cost Taylor Expansion ──────────────────────────────────────────────────────

template <typename Dims_T>
auto FinalCostTaylorExpansion<Dims_T>::operator+=(const FinalCostTaylorExpansion& rhs)
    -> FinalCostTaylorExpansion&
{
    lf += rhs.lf;
    lf_x += rhs.lf_x;
    lf_xx += rhs.lf_xx;
    return *this;
}

}  // namespace ilqr
