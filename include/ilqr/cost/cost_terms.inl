#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

#include "ilqr/cost/cost_terms.hpp"

namespace ilqr
{

// ─── Quadratic State Regulator Cost ──────────────────────────────────────────────

template <typename Dims_T>
QuadraticStateRegulatorCost<Dims_T>::QuadraticStateRegulatorCost(const StateMat& Q,
                                                                 const StateVec& x_ref)
    : Q_(Q), x_ref_(x_ref)
{
    assert(Q_.isApprox(Q_.transpose()) && "Q must be symmetric");
}

template <typename Dims_T>
auto QuadraticStateRegulatorCost<Dims_T>::evaluate(const StateVec& x, const ControlVec&, int) const
    -> Scalar
{
    const auto dx = x - x_ref_;
    return Scalar(0.5) * dx.dot(Q_ * dx);
}

template <typename Dims_T>
auto QuadraticStateRegulatorCost<Dims_T>::quadratize(const StateVec& x, const ControlVec&,
                                                     int) const -> CostTaylorExpansion<Dims_T>
{
    auto cost_taylor_expansion = CostTaylorExpansion<Dims_T>{};
    const auto dx = x - x_ref_;
    cost_taylor_expansion.l = Scalar(0.5) * dx.dot(Q_ * dx);
    cost_taylor_expansion.l_x = Q_ * dx;
    cost_taylor_expansion.l_xx = Q_;
    return cost_taylor_expansion;
}

template <typename Dims_T>
auto QuadraticStateRegulatorCost<Dims_T>::evaluate_final(const StateVec&) const -> Scalar
{
    return Scalar(0);
}

template <typename Dims_T>
auto QuadraticStateRegulatorCost<Dims_T>::quadratize_final(const StateVec&) const
    -> FinalCostTaylorExpansion<Dims_T>
{
    return FinalCostTaylorExpansion<Dims_T>{};
}

// ─── Quatratic Tracking Cost ──────────────────────────────────────────────────────

template <typename Dims_T>
QuadraticTrackingCost<Dims_T>::QuadraticTrackingCost(const StateMat& Q, AlignedVec<StateVec> x_ref)
    : Q_(Q), x_ref_(std::move(x_ref))
{
    assert(!x_ref_.empty() && "x_ref must have at least one reference state");
    assert(Q_.isApprox(Q_.transpose()) && "Q must be symmetric");
}

template <typename Dims_T>
auto QuadraticTrackingCost<Dims_T>::evaluate(const StateVec& x, const ControlVec&, int k) const
    -> Scalar
{
    assert(k >= 0 && static_cast<std::size_t>(k) < x_ref_.size());
    const auto dx = x - x_ref_[k];
    return Scalar(0.5) * dx.dot(Q_ * dx);
}

template <typename Dims_T>
auto QuadraticTrackingCost<Dims_T>::quadratize(const StateVec& x, const ControlVec&, int k) const
    -> CostTaylorExpansion<Dims_T>
{
    assert(k >= 0 && static_cast<std::size_t>(k) < x_ref_.size());
    auto cost_taylor_expansion = CostTaylorExpansion<Dims_T>{};
    const auto dx = x - x_ref_[k];
    cost_taylor_expansion.l = Scalar(0.5) * dx.dot(Q_ * dx);
    cost_taylor_expansion.l_x = Q_ * dx;
    cost_taylor_expansion.l_xx = Q_;
    return cost_taylor_expansion;
}

template <typename Dims_T>
auto QuadraticTrackingCost<Dims_T>::evaluate_final(const StateVec&) const -> Scalar
{
    return Scalar(0);
}

template <typename Dims_T>
auto QuadraticTrackingCost<Dims_T>::quadratize_final(const StateVec&) const
    -> FinalCostTaylorExpansion<Dims_T>
{
    return FinalCostTaylorExpansion<Dims_T>{};
}

// ─── Time Varying Quatratic Tracking Cost ──────────────────────────────────────────────────────

template <typename Dims_T>
TimeVaryingQuadraticTrackingCost<Dims_T>::TimeVaryingQuadraticTrackingCost(
    AlignedVec<StateMat> Q, AlignedVec<StateVec> x_ref)
    : Q_(std::move(Q)), x_ref_(std::move(x_ref))
{
    assert(!x_ref_.empty() && "x_ref must have at least one reference state");
    assert(Q_.size() == x_ref_.size() && "Q and x_ref must have equal length");
}

template <typename Dims_T>
auto TimeVaryingQuadraticTrackingCost<Dims_T>::evaluate(const StateVec& x, const ControlVec&,
                                                        int k) const -> Scalar
{
    assert(k >= 0 && static_cast<std::size_t>(k) < x_ref_.size());
    const StateVec dx = x - x_ref_[k];
    return Scalar(0.5) * dx.dot(Q_[k] * dx);
}

template <typename Dims_T>
auto TimeVaryingQuadraticTrackingCost<Dims_T>::quadratize(const StateVec& x, const ControlVec&,
                                                          int k) const
    -> CostTaylorExpansion<Dims_T>
{
    assert(k >= 0 && static_cast<std::size_t>(k) < x_ref_.size());
    auto cost_taylor_expansion = CostTaylorExpansion<Dims_T>{};
    const auto dx = x - x_ref_[k];
    cost_taylor_expansion.l = Scalar(0.5) * dx.dot(Q_[k] * dx);
    cost_taylor_expansion.l_x = Q_[k] * dx;
    cost_taylor_expansion.l_xx = Q_[k];
    return cost_taylor_expansion;
}

template <typename Dims_T>
auto TimeVaryingQuadraticTrackingCost<Dims_T>::evaluate_final(const StateVec&) const -> Scalar
{
    return Scalar(0);
}

template <typename Dims_T>
auto TimeVaryingQuadraticTrackingCost<Dims_T>::quadratize_final(const StateVec&) const
    -> FinalCostTaylorExpansion<Dims_T>
{
    return FinalCostTaylorExpansion<Dims_T>{};
}

// ─── ControlPenaltyCost ──────────────────────────────────────────────────────

template <typename Dims_T>
ControlPenaltyCost<Dims_T>::ControlPenaltyCost(const ControlMat& R) : R_(R)
{
    assert(R_.isApprox(R_.transpose()) && "R must be symmetric");
}

template <typename Dims_T>
auto ControlPenaltyCost<Dims_T>::evaluate(const StateVec&, const ControlVec& u, int) const -> Scalar
{
    return Scalar(0.5) * u.dot(R_ * u);
}

template <typename Dims_T>
auto ControlPenaltyCost<Dims_T>::quadratize(const StateVec&, const ControlVec& u, int) const
    -> CostTaylorExpansion<Dims_T>
{
    auto cost_taylor_expansion = CostTaylorExpansion<Dims_T>{};
    cost_taylor_expansion.l = Scalar(0.5) * u.dot(R_ * u);
    cost_taylor_expansion.l_u = R_ * u;
    cost_taylor_expansion.l_uu = R_;
    return cost_taylor_expansion;
}

template <typename Dims_T>
auto ControlPenaltyCost<Dims_T>::evaluate_final(const StateVec&) const -> Scalar
{
    return Scalar(0);
}

template <typename Dims_T>
auto ControlPenaltyCost<Dims_T>::quadratize_final(const StateVec&) const
    -> FinalCostTaylorExpansion<Dims_T>
{
    return FinalCostTaylorExpansion<Dims_T>{};
}

// ─── TimeVaryingControlPenaltyCost ──────────────────────────────────────────────────────

template <typename Dims_T>
TimeVaryingControlPenaltyCost<Dims_T>::TimeVaryingControlPenaltyCost(AlignedVec<ControlMat> R)
    : R_(std::move(R))
{
}

template <typename Dims_T>
auto TimeVaryingControlPenaltyCost<Dims_T>::evaluate(const StateVec&, const ControlVec& u,
                                                     int k) const -> Scalar
{
    return Scalar(0.5) * u.dot(R_[k] * u);
}

template <typename Dims_T>
auto TimeVaryingControlPenaltyCost<Dims_T>::quadratize(const StateVec&, const ControlVec& u,
                                                       int k) const -> CostTaylorExpansion<Dims_T>
{
    assert(k >= 0 && static_cast<std::size_t>(k) < R_.size());
    auto cost_taylor_expansion = CostTaylorExpansion<Dims_T>{};
    cost_taylor_expansion.l = Scalar(0.5) * u.dot(R_[k] * u);
    cost_taylor_expansion.l_u = R_[k] * u;
    cost_taylor_expansion.l_uu = R_[k];
    return cost_taylor_expansion;
}

template <typename Dims_T>
auto TimeVaryingControlPenaltyCost<Dims_T>::evaluate_final(const StateVec&) const -> Scalar
{
    return Scalar(0);
}

template <typename Dims_T>
auto TimeVaryingControlPenaltyCost<Dims_T>::quadratize_final(const StateVec&) const
    -> FinalCostTaylorExpansion<Dims_T>
{
    return FinalCostTaylorExpansion<Dims_T>{};
}

// ─── FinalCost ───────────────────────────────────────────────────────────────

template <typename Dims_T>
FinalCost<Dims_T>::FinalCost(const StateMat& Qf, const StateVec& x_goal) : Qf_(Qf), x_goal_(x_goal)
{
    assert(Qf_.isApprox(Qf_.transpose()) && "Qf must be symmetric");
}

template <typename Dims_T>
auto FinalCost<Dims_T>::evaluate(const StateVec&, const ControlVec&, int) const -> Scalar
{
    return Scalar(0);
}

template <typename Dims_T>
auto FinalCost<Dims_T>::quadratize(const StateVec&, const ControlVec&, int) const
    -> CostTaylorExpansion<Dims_T>
{
    return CostTaylorExpansion<Dims_T>{};
}

template <typename Dims_T>
auto FinalCost<Dims_T>::evaluate_final(const StateVec& x) const -> Scalar
{
    const auto dx = x - x_goal_;
    return Scalar(0.5) * dx.dot(Qf_ * dx);
}

template <typename Dims_T>
auto FinalCost<Dims_T>::quadratize_final(const StateVec& x) const
    -> FinalCostTaylorExpansion<Dims_T>
{
    auto final_cost_taylor_expansion = FinalCostTaylorExpansion<Dims_T>{};
    const auto dx = x - x_goal_;
    final_cost_taylor_expansion.lf = Scalar(0.5) * dx.dot(Qf_ * dx);
    final_cost_taylor_expansion.lf_x = Qf_ * dx;
    final_cost_taylor_expansion.lf_xx = Qf_;
    return final_cost_taylor_expansion;
}

}  // namespace ilqr
