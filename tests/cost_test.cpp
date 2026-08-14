// Unit tests for the cost expansions, term library, and compile-time composite.

#include "ilqr/cost/composite_cost.hpp"

#include <gtest/gtest.h>

#include "ilqr/cost/cost_terms.hpp"

namespace
{
using Dims = ilqr::Dims<2, 1>;
using StateMat = Dims::StateMat;
using ControlMat = Dims::ControlMat;
using StateVec = Dims::StateVec;
using ControlVec = Dims::ControlVec;

TEST(Cost, cost)
{
    const auto zero_state = StateVec::Zero();
    const auto zero_controls = ControlVec::Zero();
    auto const Q = StateMat::Identity();

    ilqr::AlignedVec<StateVec> const x_ref{zero_state, zero_state};
    auto const quadratic_cost = ilqr::QuadraticTrackingCost<Dims>(Q, std::move(x_ref));

    EXPECT_EQ(quadratic_cost.evaluate(zero_state, zero_controls, 0), 0.0);
}

TEST(CompositeCost, composite)
{
    const auto zero_state = StateVec::Zero();
    const auto zero_controls = ControlVec::Zero();

    auto const Qf = StateMat::Identity();
    auto const R = ControlMat::Identity();

    auto const control_input_penalty_cost = ilqr::ControlPenaltyCost<Dims>(R);
    auto const final_cost = ilqr::FinalCost<Dims>(Qf, StateVec::Zero());

    auto const composite_cost = ilqr::CompositeCostFunction(control_input_penalty_cost, final_cost);

    EXPECT_EQ(composite_cost.evaluate(zero_state, zero_controls, 0), 0.0);
}

TEST(QuadraticStateRegulatorCost, RegulatesTowardReference)
{
    static_assert(ilqr::CostFunction<ilqr::QuadraticStateRegulatorCost<Dims>>,
                  "QuadraticStateRegulatorCost must satisfy CostFunction");

    const auto Q = (StateMat() << 2.0, 0.0, 0.0, 4.0).finished();
    const StateVec x = (StateVec() << 3.0, 2.0).finished();
    const ControlVec u = ControlVec::Zero();

    // Default zero reference: dx = x, ½ dxᵀ Q dx = ½ (2·9 + 4·4) = 17
    const ilqr::QuadraticStateRegulatorCost<Dims> regulator(Q);
    EXPECT_DOUBLE_EQ(regulator.evaluate(x, u, 0), 17.0);
    const auto e = regulator.quadratize(x, u, 0);
    EXPECT_DOUBLE_EQ(e.l, 17.0);
    EXPECT_DOUBLE_EQ(e.l_x(0), 6.0);  // Q·x = [2·3, 4·2]
    EXPECT_DOUBLE_EQ(e.l_x(1), 8.0);
    EXPECT_DOUBLE_EQ(e.l_xx(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(e.l_xx(1, 1), 4.0);

    // Running-only: no final cost.
    EXPECT_DOUBLE_EQ(regulator.evaluate_final(x), 0.0);
    EXPECT_TRUE(regulator.quadratize_final(x).lf_xx.isZero());

    // Constant setpoint: dx = [2, 3], ½ (2·4 + 4·9) = 22
    const StateVec x_ref = (StateVec() << 1.0, -1.0).finished();
    const ilqr::QuadraticStateRegulatorCost<Dims> setpoint(Q, x_ref);
    EXPECT_DOUBLE_EQ(setpoint.evaluate(x, u, 0), 22.0);
}
/*
TEST(CostTaylorExpansion, SetZeroAndAccumulate)
{
    ilqr::CostTaylorExpansion<Dims> a;
    a.setZero();
    EXPECT_EQ(a.l, 0.0);
    EXPECT_TRUE(a.lx.isZero());
    EXPECT_TRUE(a.lxx.isZero());

    ilqr::CostTaylorExpansion<Dims> b;
    b.setZero();
    b.l = 2.0;
    b.lx << 1.0, 3.0;
    b.lxx = Dims::StateMat::Identity();

    a += b;
    a += b;
    EXPECT_DOUBLE_EQ(a.l, 4.0);
    EXPECT_DOUBLE_EQ(a.lx(0), 2.0);
    EXPECT_DOUBLE_EQ(a.lx(1), 6.0);
    EXPECT_DOUBLE_EQ(a.lxx(0, 0), 2.0);
}

TEST(QuadraticTrackingCost, MatchesHandComputation)
{
    ilqr::QuadraticTrackingCost<Dims> track;
    track.Q = (Dims::StateMat() << 2.0, 0.0, 0.0, 4.0).finished();
    track.x_ref.assign(3, StateVec::Zero());
    track.x_ref[1] << 1.0, -1.0;

    const StateVec   x = (StateVec() << 3.0, 2.0).finished();
    const ControlVec u = ControlVec::Zero();

    // dx = [2, 3], ½ dxᵀ Q dx = ½ (2·4 + 4·9) = ½·44 = 22
    const auto e = track.quadratize(x, u, 1);
    EXPECT_NEAR(e.l, 22.0, kTol);
    EXPECT_NEAR(e.lx(0), 4.0, kTol);   // Q·dx = [2·2, 4·3]
    EXPECT_NEAR(e.lx(1), 12.0, kTol);
    EXPECT_NEAR(e.lxx(0, 0), 2.0, kTol);
    EXPECT_NEAR(e.lxx(1, 1), 4.0, kTol);
    EXPECT_NEAR(track.evaluate(x, u, 1), 22.0, kTol);

    // Tracking is running-only.
    EXPECT_NEAR(track.evaluate_final(x), 0.0, kTol);
    EXPECT_TRUE(track.quadratize_final(x).lxx.isZero());
}

TEST(CompositeCostFunction, SumsTermsAndSatisfiesConcept)
{
    ilqr::ControlPenaltyCost<Dims> u_cost;
    u_cost.R = (Dims::ControlMat() << 0.5).finished();

    ilqr::FinalCost<Dims> term;
    term.Qf     = Dims::StateMat::Identity();
    term.x_goal << 1.0, 0.0;

    auto composite = ilqr::CompositeCostFunction{u_cost, term};
    static_assert(ilqr::CostFunction<decltype(composite)>,
                  "the composite must itself satisfy CostFunction");

    const StateVec   x = (StateVec() << 3.0, 0.0).finished();
    const ControlVec u = (ControlVec() << 2.0).finished();

    // Running: only the control term. ½ · 0.5 · 2² = 1.0
    EXPECT_NEAR(composite.evaluate(x, u, 0), 1.0, kTol);
    EXPECT_NEAR(composite.quadratize(x, u, 0).luu(0, 0), 0.5, kTol);
    EXPECT_NEAR(composite.quadratize(x, u, 0).l, 1.0, kTol);

    // Final: only the final term. dx = [2, 0], ½·(2²) = 2.0
    EXPECT_NEAR(composite.evaluate_final(x), 2.0, kTol);
    EXPECT_NEAR(composite.quadratize_final(x).l, 2.0, kTol);
    EXPECT_NEAR(composite.quadratize_final(x).lx(0), 2.0, kTol);

    // Composite == sum of individual terms.
    EXPECT_NEAR(composite.evaluate(x, u, 0),
                u_cost.evaluate(x, u, 0) + term.evaluate(x, u, 0), kTol);
    EXPECT_NEAR(composite.evaluate_final(x),
                u_cost.evaluate_final(x) + term.evaluate_final(x), kTol);
}
*/
}  // namespace
