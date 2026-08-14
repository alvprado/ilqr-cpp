#pragma once

#include <Eigen/Dense>

#include "ilqr/core/types.hpp"
#include "ilqr/cost/cost_concepts.hpp"

namespace ilqr
{

/// @brief Running cost ½ (x − x_ref)ᵀ Q (x − x_ref) toward a constant reference; zero final cost.
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
class QuadraticStateRegulatorCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using StateMat = typename Dims_T::StateMat;

    /// @brief Builds the term from a constant weight and an optional constant reference.
    /// @param Q State weighting matrix (expected symmetric positive semi-definite).
    /// @param x_ref Reference state to regulate toward; defaults to zero.
    explicit QuadraticStateRegulatorCost(const StateMat& Q,
                                         const StateVec& x_ref = StateVec::Zero());

    /// @brief Evaluates the running state cost at x.
    /// @param x The state at timestep k
    /// @param u The control (unused; the cost penalizes state only)
    /// @param k The timestep index (unused; the cost is time-invariant)
    /// @return ½ (x − x_ref)ᵀ Q (x − x_ref)
    Scalar evaluate(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Builds the running-cost quadratic expansion at x.
    /// @param x The state at timestep k
    /// @param u The control (unused)
    /// @param k The timestep index (unused)
    /// @return Expansion with l, l_x and l_xx set (control terms are zero)
    CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Final cost of this term; always zero (this is a running cost).
    /// @param x The final state (unused)
    /// @return Scalar(0)
    Scalar evaluate_final(const StateVec& x) const;

    /// @brief Final-cost expansion of this term; always zero.
    /// @param x The final state (unused)
    /// @return A zero-initialized FinalCostTaylorExpansion
    FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const;

private:
    /// Weight matrix
    StateMat Q_;
    /// Reference state
    StateVec x_ref_;
};

/// @brief Running cost ½ (x − x_ref[k])ᵀ Q (x − x_ref[k]) with a constant weight
///        matrix and a per-timestep reference; zero final cost.
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
class QuadraticTrackingCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using StateMat = typename Dims_T::StateMat;

    /// @brief Builds the term from a constant weight and a per-timestep reference.
    /// @param Q State weighting matrix (expected symmetric positive semi-definite).
    /// @param x_ref Reference states; x_ref[k] is the target at running step k. Must be
    ///        non-empty and cover every running step the solver evaluates.
    QuadraticTrackingCost(const StateMat& Q, AlignedVec<StateVec> x_ref);

    /// @brief Evaluates the running tracking cost at (x, k).
    /// @param x The state at timestep k
    /// @param u The control (unused; tracking penalizes state only)
    /// @param k The timestep index, selecting the reference x_ref[k]
    /// @return ½ (x − x_ref[k])ᵀ Q (x − x_ref[k])
    Scalar evaluate(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Builds the running-cost quadratic expansion at (x, k).
    /// @param x The state at timestep k
    /// @param u The control (unused)
    /// @param k The timestep index, selecting the reference x_ref[k]
    /// @return Taylor Expansion with l, l_x and l_xx set (control terms are zero)
    CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Final cost of this term; always zero (tracking is a running cost).
    /// @param x The final state (unused)
    /// @return Scalar(0)
    Scalar evaluate_final(const StateVec& x) const;

    /// @brief Final-cost expansion of this term; always zero.
    /// @param x The final state (unused)
    /// @return A zero-initialized FinalCostTaylorExpansion
    FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const;

private:
    /// Weight matrix
    StateMat Q_;
    /// Reference states
    AlignedVec<StateVec> x_ref_;
};

/// @brief Running cost ½ (x − x_ref[k])ᵀ Q[k] (x − x_ref[k]) with a per-timestep
///        weight matrix and reference; zero final cost.
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
class TimeVaryingQuadraticTrackingCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using StateMat = typename Dims_T::StateMat;

    /// @brief Builds the term from per-timestep weights and references.
    /// @param Q Weighting matrices; Q[k] is applied at running step k (each expected
    ///        symmetric positive semi-definite).
    /// @param x_ref Reference states; x_ref[k] is the target at running step k.
    /// @note Q and x_ref must be non-empty and of equal length.
    TimeVaryingQuadraticTrackingCost(AlignedVec<StateMat> Q, AlignedVec<StateVec> x_ref);

    /// @brief Evaluates the running tracking cost at (x, k) using the step-k weight.
    /// @param x The state at timestep k
    /// @param u The control (unused; tracking penalizes state only)
    /// @param k The timestep index, selecting Q[k] and x_ref[k]
    /// @return ½ (x − x_ref[k])ᵀ Q[k] (x − x_ref[k])
    Scalar evaluate(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Builds the running-cost quadratic expansion at (x, k) using Q[k].
    /// @param x The state at timestep k
    /// @param u The control (unused)
    /// @param k The timestep index, selecting Q[k] and x_ref[k]
    /// @return Expansion with l, l_x and l_xx set (control terms zero)
    CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Final cost of this term; always zero (tracking is a running cost).
    /// @param x The final state (unused)
    /// @return Scalar(0)
    Scalar evaluate_final(const StateVec& x) const;

    /// @brief Final-cost expansion of this term; always zero.
    /// @param x The final state (unused)
    /// @return A zero-initialized FinalCostTaylorExpansion
    FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const;

private:
    /// Vector of weight matrixes
    AlignedVec<StateMat> Q_;
    /// Vector of reference states
    AlignedVec<StateVec> x_ref_;
};

/// @brief Running cost ½ uᵀ R u; zero final cost.
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
class ControlPenaltyCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using ControlMat = typename Dims_T::ControlMat;

    /// @brief Builds the term from a control weight and an optional reference.
    /// @param R Control weighting matrix (expected symmetric positive semi-definite).
    /// @param u_ref Reference control; defaults to zero (a pure effort penalty).
    explicit ControlPenaltyCost(const ControlMat& R);

    /// @brief Evaluates the running control-effort cost at u.
    /// @param x The state (unused; the penalty is on control only)
    /// @param u The control at this timestep
    /// @param k The timestep index (unused; the penalty is time-invariant)
    /// @return ½ uᵀ R u
    Scalar evaluate(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Builds the running-cost quadratic expansion at u.
    /// @param x The state (unused)
    /// @param u The control at this timestep
    /// @param k The timestep index (unused)
    /// @return Expansion with l, l_u and l_uu set (state terms zero)
    CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Final cost of this term; always zero (this is a running cost).
    /// @param x The final state (unused)
    /// @return Scalar(0)
    Scalar evaluate_final(const StateVec& x) const;

    /// @brief Final-cost expansion of this term; always zero.
    /// @param x The final state (unused)
    /// @return A zero-initialized FinalCostTaylorExpansion
    FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const;

private:
    ControlMat R_;
};

/// @brief Running cost ½ uᵀ R_k u; zero final cost with a per-timestep
///        weight matrix
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
class TimeVaryingControlPenaltyCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using ControlMat = typename Dims_T::ControlMat;

    /// @brief Builds the term from a control weight and an optional reference.
    /// @param R Control weighting matrix (expected symmetric positive semi-definite).
    /// @param u_ref Reference control; defaults to zero (a pure effort penalty).
    explicit TimeVaryingControlPenaltyCost(AlignedVec<ControlMat> R);

    /// @brief Evaluates the running control-effort cost at u.
    /// @param x The state (unused; the penalty is on control only)
    /// @param u The control at this timestep
    /// @param k The timestep index
    /// @return ½ uᵀ R_k u
    Scalar evaluate(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Builds the running-cost quadratic expansion at u.
    /// @param x The state (unused)
    /// @param u The control at this timestep
    /// @param k The timestep index
    /// @return Expansion with l, l_u and l_uu set (state terms zero) at timestep k
    CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Final cost of this term; always zero (this is a running cost).
    /// @param x The final state (unused)
    /// @return Scalar(0)
    Scalar evaluate_final(const StateVec& x) const;

    /// @brief Final-cost expansion of this term; always zero.
    /// @param x The final state (unused)
    /// @return A zero-initialized FinalCostTaylorExpansion
    FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const;

private:
    AlignedVec<ControlMat> R_;
};

/// @brief Final cost ½ (x − x_goal)ᵀ Qf (x − x_goal); zero running cost.
/// @tparam Dims_T The dimensions the cost operates on
template <typename Dims_T>
class FinalCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using StateMat = typename Dims_T::StateMat;

    /// @brief Builds the term from a final weight and a goal state.
    /// @param Qf Final state weighting matrix (expected symmetric positive semi-definite).
    /// @param x_goal Goal state.
    FinalCost(const StateMat& Qf, const StateVec& x_goal);

    /// @brief Running cost of this term; always zero (this is a final cost).
    /// @param x The state (unused)
    /// @param u The control (unused)
    /// @param k The timestep index (unused)
    /// @return Scalar(0)
    Scalar evaluate(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Running-cost expansion of this term; always zero.
    /// @param x The state (unused)
    /// @param u The control (unused)
    /// @param k The timestep index (unused)
    /// @return A zero-initialized CostTaylorExpansion
    CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec& u, int k) const;

    /// @brief Evaluates the final cost at the terminal state x_N.
    /// @param x The final state x_N
    /// @return ½ (x − x_goal)ᵀ Qf (x − x_goal)
    Scalar evaluate_final(const StateVec& x) const;

    /// @brief Builds the final-cost quadratic expansion at x_N.
    /// @param x The final state x_N
    /// @return Expansion with lf, lf_x and lf_xx set
    FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const;

private:
    StateMat Qf_;
    StateVec x_goal_;
};

}  // namespace ilqr

#include "ilqr/cost/cost_terms.inl"
