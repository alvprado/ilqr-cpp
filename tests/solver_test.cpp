// Solver tests. The key anchor: for linear dynamics + quadratic cost, iLQR must
// reproduce the exact finite-horizon LQR (Riccati) solution — matching both the
// optimal cost and the per-step feedback gains — and converge almost immediately.

#include "ilqr/core/solver.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "ilqr/cost/composite_cost.hpp"
#include "ilqr/cost/cost_terms.hpp"
#include "ilqr/dynamics/dynamics_concepts.hpp"
#include "ilqr/core/solver_diagnostics.hpp"
#include "ilqr/core/solver_io.hpp"
#include "ilqr/core/trajectory.hpp"
#include "ilqr/core/types.hpp"

namespace
{
using Dims = ilqr::Dims<2, 1>;
using StateVec = Dims::StateVec;
using ControlVec = Dims::ControlVec;
using StateMat = Dims::StateMat;
using ControlMat = Dims::ControlMat;
using StateControlMat = Dims::StateControlMat;
using ControlStateMat = Dims::ControlStateMat;

// A time-invariant linear system x_{k+1} = A x + B u.
struct LinearDynamics
{
    using Dims = ilqr::Dims<2, 1>;
    StateMat A;
    StateControlMat B;

    StateVec step(const StateVec& x, const ControlVec& u) const { return A * x + B * u; }
    ilqr::DynamicsTaylorExpansion<Dims> linearize(const StateVec&, const ControlVec&) const
    {
        return {A, B};
    }
};

// Total cost of a trajectory under a given cost function (the solver keeps this private,
// so the test recomputes it to compare against the analytic LQR optimum).
template <typename Cost_T>
double trajectory_cost(const Cost_T& cost, const ilqr::Trajectory<Dims>& trajectory)
{
    const int N = trajectory.horizon();
    double total = cost.evaluate_final(trajectory.state(N));
    for (int k = 0; k < N; ++k)
        total += cost.evaluate(trajectory.state(k), trajectory.control(k), k);
    return total;
}

TEST(SolverLQR, MatchesRiccatiSolution)
{
    // Double-integrator-like linear system.
    StateMat A;
    A << 1.0, 0.1, 0.0, 1.0;
    StateControlMat B;
    B << 0.0, 0.1;

    const StateMat Q = 0.1 * StateMat::Identity();
    const ControlMat R = (ControlMat() << 0.01).finished();
    const StateMat Qf = StateMat::Identity();
    const int N = 30;

    LinearDynamics dyn{A, B};

    // Running ½xᵀQx + ½uᵀRu (tracking to the origin), final ½xᵀQf x.
    ilqr::AlignedVec<StateVec> x_ref(static_cast<std::size_t>(N), StateVec::Zero());
    ilqr::QuadraticTrackingCost<Dims> state_cost(Q, x_ref);
    ilqr::ControlPenaltyCost<Dims> ctrl_cost(R);
    ilqr::FinalCost<Dims> term_cost(Qf, StateVec::Zero());

    auto cost = ilqr::CompositeCostFunction{state_cost, ctrl_cost, term_cost};

    ilqr::SolverConfig<double> opts;
    opts.regularization.init = 0.0;  // linear-quadratic: Q_uu is always PD, so no reg needed
    opts.regularization.min = 0.0;   // → gains match the exact (unregularized) Riccati solution

    ilqr::ILQRSolver solver{dyn, cost, opts};

    StateVec x0;
    x0 << 1.0, 0.0;
    const auto request = ilqr::SolveRequest<Dims>::cold_start(x0, N);
    const auto result = solver.solve(request);

    // --- Independent reference: finite-horizon discrete LQR via Riccati. ---
    ControlStateMat Klqr[30];  // N == 30
    StateMat P = Qf;
    for (int k = N - 1; k >= 0; --k)
    {
        const ControlMat S = R + B.transpose() * P * B;
        Klqr[k] = S.inverse() * (B.transpose() * P * A);
        P = Q + A.transpose() * P * A - A.transpose() * P * B * Klqr[k];
    }
    const double lqr_cost = 0.5 * (x0.transpose() * P * x0)(0, 0);

    // --- Assertions. ---
    EXPECT_EQ(result.status, ilqr::SolverStatus::Converged);
    EXPECT_NEAR(trajectory_cost(cost, result.trajectory), lqr_cost, 1e-9);

    // iLQR feedback gain K satisfies du = K·dx, i.e. K == -K_lqr.
    ASSERT_EQ(static_cast<int>(result.feedback_gains.size()), N);
    for (int k = 0; k < N; ++k)
        EXPECT_TRUE(result.feedback_gains[static_cast<std::size_t>(k)].isApprox(-Klqr[k], 1e-7))
            << "gain mismatch at k=" << k;
}

TEST(SolverDiagnostics, RecordsPerIteration)
{
    // Same LQ problem as above, but solved through the diagnostics overload.
    StateMat A;
    A << 1.0, 0.1, 0.0, 1.0;
    StateControlMat B;
    B << 0.0, 0.1;

    const StateMat Q = 0.1 * StateMat::Identity();
    const ControlMat R = (ControlMat() << 0.01).finished();
    const StateMat Qf = StateMat::Identity();
    const int N = 30;

    LinearDynamics dyn{A, B};
    ilqr::AlignedVec<StateVec> x_ref(static_cast<std::size_t>(N), StateVec::Zero());
    ilqr::QuadraticTrackingCost<Dims> state_cost(Q, x_ref);
    ilqr::ControlPenaltyCost<Dims> ctrl_cost(R);
    ilqr::FinalCost<Dims> term_cost(Qf, StateVec::Zero());
    auto cost = ilqr::CompositeCostFunction{state_cost, ctrl_cost, term_cost};

    ilqr::SolverConfig<double> opts;
    opts.regularization.init = 0.0;
    opts.regularization.min = 0.0;

    ilqr::ILQRSolver solver{dyn, cost, opts};

    StateVec x0;
    x0 << 1.0, 0.0;
    const auto request = ilqr::SolveRequest<Dims>::cold_start(x0, N);

    ilqr::SolveDiagnostics<double> diagnostics;
    const auto result = solver.solve(request, diagnostics);

    EXPECT_EQ(result.status, ilqr::SolverStatus::Converged);

    // One record per iteration, with at least one accepted step.
    ASSERT_FALSE(diagnostics.iterations.empty());
    const int accepted_count = static_cast<int>(
        std::count_if(diagnostics.iterations.begin(), diagnostics.iterations.end(),
                      [](const auto& record) { return record.accepted_forward_pass; }));
    EXPECT_GE(accepted_count, 1);

    // Per-record invariants: reg always recorded; accepted steps carry a valid step size and a
    // non-increasing cost; non-accepted iterations leave step_size as NaN.
    double previous_accepted_cost = std::numeric_limits<double>::infinity();
    for (const auto& record : diagnostics.iterations)
    {
        EXPECT_TRUE(std::isfinite(record.reg));
        EXPECT_GE(record.reg, 0.0);
        if (record.accepted_forward_pass)
        {
            EXPECT_TRUE(std::isfinite(record.step_size));
            EXPECT_GT(record.step_size, 0.0);
            EXPECT_LE(record.step_size, 1.0);
            EXPECT_LE(record.cost, previous_accepted_cost + 1e-12);
            previous_accepted_cost = record.cost;
        }
        else
        {
            EXPECT_TRUE(std::isnan(record.step_size));
        }
    }

    // The last accepted cost equals the converged trajectory cost (== analytic LQR optimum,
    // verified in MatchesRiccatiSolution).
    EXPECT_NEAR(previous_accepted_cost, trajectory_cost(cost, result.trajectory), 1e-9);

    // This LQ problem converges via the gradient-norm test, so the final record is a
    // non-accepted iteration whose gradient norm is below tolerance.
    const auto& last_record = diagnostics.iterations.back();
    if (!last_record.accepted_forward_pass)
        EXPECT_LT(last_record.grad_norm, opts.convergence.grad_tol);

    // Reusing the same diagnostics object clears it on entry: a second solve does not accumulate.
    const std::size_t records_after_first_solve = diagnostics.iterations.size();
    const auto reused_result = solver.solve(request, diagnostics);
    EXPECT_EQ(reused_result.status, ilqr::SolverStatus::Converged);
    EXPECT_EQ(diagnostics.iterations.size(), records_after_first_solve);
}
}  // namespace
