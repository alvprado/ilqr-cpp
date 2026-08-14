// Tests for the continuous-model discretization + finite-difference Jacobian machinery:
//   - finite-difference Jacobians match the analytic discretization on a linear model,
//   - the integrators have the expected accuracy ordering (RK4 < Heun < Euler error),
//   - a continuous model solves end to end through discretize() + iLQR.
// (The AutoDiff policy is covered separately in autodiff_test.cpp.)

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "ilqr/ilqr.hpp"

namespace
{

// x_dot = Ac x + Bc u  (linear, so central finite differences of any explicit integrator are exact).
struct LinearContinuous
{
    using Dims = ilqr::Dims<2, 1>;
    Dims::StateMat Ac;
    Dims::StateControlMat Bc;
    Dims::StateVec operator()(const Dims::StateVec& x, const Dims::ControlVec& u) const
    {
        return Ac * x + Bc * u;
    }
};

// x_dot = -x  (scalar decay; exact solution x(dt) = x0 e^{-dt}). Control is unused.
struct Decay
{
    using Dims = ilqr::Dims<1, 1>;
    Dims::StateVec operator()(const Dims::StateVec& x, const Dims::ControlVec&) const { return -x; }
};

// x_dot = [vel, u]  (double integrator as a continuous model).
struct DoubleIntegratorContinuous
{
    using Dims = ilqr::Dims<2, 1>;
    Dims::StateVec operator()(const Dims::StateVec& x, const Dims::ControlVec& u) const
    {
        Dims::StateVec d;
        d << x(1), u(0);
        return d;
    }
};

TEST(FiniteDifference, MatchesAnalyticEulerJacobiansOnLinearModel)
{
    using Dims = ilqr::Dims<2, 1>;
    Dims::StateMat Ac;
    Ac << 0.0, 1.0, -2.0, -0.5;
    Dims::StateControlMat Bc;
    Bc << 0.0, 1.0;
    const double dt = 0.01;

    auto dyn = ilqr::discretize<Dims>(LinearContinuous{Ac, Bc}, dt, ilqr::math::EulerStep{});

    Dims::StateVec x;
    x << 0.3, -0.2;
    Dims::ControlVec u;
    u << 0.5;

    const auto expansion = dyn.linearize(x, u);

    // Euler on a linear model: A_d = I + dt Ac,  B_d = dt Bc.
    const Dims::StateMat A_expected = Dims::StateMat::Identity() + dt * Ac;
    const Dims::StateControlMat B_expected = dt * Bc;

    EXPECT_TRUE(expansion.A.isApprox(A_expected, 1e-6)) << expansion.A;
    EXPECT_TRUE(expansion.B.isApprox(B_expected, 1e-6)) << expansion.B;
}

TEST(Integrators, AccuracyOrderingRK4BetterThanHeunBetterThanEuler)
{
    Decay model;
    Decay::Dims::StateVec x0;
    x0 << 1.0;
    Decay::Dims::ControlVec u;
    u << 0.0;
    const double dt = 0.5;
    const double exact = std::exp(-dt);

    const double euler = ilqr::math::EulerStep{}(model, x0, u, dt)(0);
    const double heun = ilqr::math::HeunStep{}(model, x0, u, dt)(0);
    const double rk4 = ilqr::math::RK4Step{}(model, x0, u, dt)(0);

    EXPECT_LT(std::abs(rk4 - exact), std::abs(heun - exact));
    EXPECT_LT(std::abs(heun - exact), std::abs(euler - exact));
}

TEST(DiscretizedModel, SolvesContinuousModelEndToEnd)
{
    using Dims = ilqr::Dims<2, 1>;
    using StateVec = Dims::StateVec;
    using ControlVec = Dims::ControlVec;
    using StateMat = Dims::StateMat;
    using ControlMat = Dims::ControlMat;

    auto dyn = ilqr::discretize<Dims>(DoubleIntegratorContinuous{}, 0.05, ilqr::math::RK4Step{});

    const ControlMat R = (ControlMat() << 0.01).finished();
    ilqr::ControlPenaltyCost<Dims> ctrl_cost(R);
    StateMat Qf = StateMat::Zero();
    Qf.diagonal() << 100.0, 100.0;
    StateVec goal;
    goal << 1.0, 0.0;
    ilqr::FinalCost<Dims> term_cost(Qf, goal);
    auto cost = ilqr::CompositeCostFunction{ctrl_cost, term_cost};

    ilqr::SolverConfig<double> config;
    ilqr::ILQRSolver solver{dyn, cost, config};

    StateVec x0;
    x0 << 0.0, 0.0;
    const auto request = ilqr::SolveRequest<Dims>::cold_start(x0, 50);

    ilqr::SolveDiagnostics<double> diagnostics;
    const auto result = solver.solve(request, diagnostics);

    EXPECT_EQ(result.status, ilqr::SolverStatus::Converged);

    // Cost is non-increasing across accepted steps.
    double previous_cost = std::numeric_limits<double>::infinity();
    for (const auto& record : diagnostics.iterations)
        if (record.accepted_forward_pass)
        {
            EXPECT_LE(record.cost, previous_cost + 1e-9);
            previous_cost = record.cost;
        }

    // The controlled trajectory reaches the goal position.
    const StateVec x_final = result.trajectory.state(result.trajectory.horizon());
    EXPECT_NEAR(x_final(0), 1.0, 1e-2);
    EXPECT_NEAR(x_final(1), 0.0, 1e-2);
}

}  // namespace
