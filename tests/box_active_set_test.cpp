// Tests for the Projected-Newton box-constrained QP solver:
//   - it reproduces the unconstrained minimizer when no bound is active,
//   - it lands on the analytic solution when bounds are active, and identifies the right free set,
//   - random problems satisfy the KKT conditions and are never worse than a projected-gradient
//     reference solve,
//   - warm starting from a previous solution cuts the iteration count,
//   - a single projected step activates many bounds at once (the property that separates this from
//     a classical ratio-test active-set method),
//   - an indefinite free Hessian is reported as std::nullopt.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>

#include "ilqr/ilqr.hpp"

namespace
{

using ilqr::math::BoxQPActiveSetMethod;
using ilqr::math::BoxQPConfig;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// KKT residual of the box QP, computed independently of the solver's own free set: the gradient on
// the interior, and only the component pointing out of the box at an active bound.
template <int N>
double kkt_residual(const Eigen::Matrix<double, N, N>& hessian,
                    const Eigen::Vector<double, N>& gradient, const Eigen::Vector<double, N>& lower,
                    const Eigen::Vector<double, N>& upper, const Eigen::Vector<double, N>& x)
{
    const Eigen::Vector<double, N> current_gradient = gradient + hessian * x;
    double residual = 0.0;

    for (int component = 0; component < N; ++component)
    {
        double projected = current_gradient(component);
        if (x(component) <= lower(component))
        {
            projected = std::min(projected, 0.0);  // multiplier at the lower bound must be >= 0
        }
        if (x(component) >= upper(component))
        {
            projected = std::max(projected, 0.0);  // multiplier at the upper bound must be >= 0
        }
        residual = std::max(residual, std::abs(projected));
    }
    return residual;
}

template <int N>
double cost(const Eigen::Matrix<double, N, N>& hessian,
            const Eigen::Vector<double, N>& gradient, const Eigen::Vector<double, N>& x)
{
    return x.dot(gradient) + 0.5 * x.dot(hessian * x);
}

// Independent reference: projected gradient descent at the fixed step 1/L. Slow, but for a strictly
// convex QP it converges to the same unique global minimizer.
template <int N>
Eigen::Vector<double, N> projected_gradient_reference(const Eigen::Matrix<double, N, N>& hessian,
                                                      const Eigen::Vector<double, N>& gradient,
                                                      const Eigen::Vector<double, N>& lower,
                                                      const Eigen::Vector<double, N>& upper)
{
    const double lipschitz_constant =
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>>(hessian).eigenvalues().maxCoeff();

    Eigen::Vector<double, N> x = Eigen::Vector<double, N>::Zero().cwiseMax(lower).cwiseMin(upper);
    for (int step = 0; step < 50000; ++step)
    {
        x = (x - (gradient + hessian * x) / lipschitz_constant).cwiseMax(lower).cwiseMin(upper);
    }
    return x;
}

TEST(BoxQPActiveSet, MatchesUnconstrainedMinimizerWhenNoBoundIsActive)
{
    Eigen::Matrix2d hessian;
    hessian << 2.0, 0.5, 0.5, 1.0;
    const Eigen::Vector2d gradient(-1.0, -4.0);
    const Eigen::Vector2d lower = Eigen::Vector2d::Constant(-100.0);
    const Eigen::Vector2d upper = Eigen::Vector2d::Constant(100.0);

    BoxQPActiveSetMethod<2> solver;
    const auto result = solver.solve(hessian, gradient, lower, upper);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->optimal_vector.isApprox(hessian.ldlt().solve(-gradient), 1e-10));
    EXPECT_TRUE((result->free_indices == Eigen::Array2i(0, 1)).all());
}

TEST(BoxQPActiveSet, MatchesAnalyticSolutionWithOneActiveBound)
{
    // Unconstrained minimizer is (-4/7, 30/7); the upper bound on the second component binds, and
    // minimizing over the first with x1 = 2 fixed gives x0 = 0.
    Eigen::Matrix2d hessian;
    hessian << 2.0, 0.5, 0.5, 1.0;
    const Eigen::Vector2d gradient(-1.0, -4.0);
    const Eigen::Vector2d lower(-10.0, -10.0);
    const Eigen::Vector2d upper(10.0, 2.0);

    BoxQPActiveSetMethod<2> solver;
    const auto result = solver.solve(hessian, gradient, lower, upper);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->optimal_vector(0), 0.0, 1e-10);
    EXPECT_NEAR(result->optimal_vector(1), 2.0, 1e-10);
    // Only the first component is free; the second is pinned to its upper bound.
    ASSERT_EQ(result->free_indices.size(), 1);
    EXPECT_EQ(result->free_indices(0), 0);
}

TEST(BoxQPActiveSet, SolvesACornerOptimumWithEveryBoundActive)
{
    const Eigen::Matrix2d hessian = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d gradient(-5.0, -5.0);
    const Eigen::Vector2d lower(0.0, 0.0);
    const Eigen::Vector2d upper(1.0, 1.0);

    BoxQPActiveSetMethod<2> solver;
    const auto result = solver.solve(hessian, gradient, lower, upper);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->optimal_vector.isApprox(upper, 1e-12));
    EXPECT_EQ(result->free_indices.size(), 0);
}

TEST(BoxQPActiveSet, ACornerOptimumCarriesNoFactorization)
{
    const Eigen::Matrix2d hessian = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d lower(0.0, 0.0);
    const Eigen::Vector2d upper(1.0, 1.0);

    BoxQPActiveSetMethod<2> solver;

    // An interior optimum leaves a 2x2 factorization cached on the solver.
    const auto interior_result = solver.solve(hessian, Eigen::Vector2d(-0.5, -0.5), lower, upper);
    ASSERT_TRUE(interior_result.has_value());
    ASSERT_EQ(interior_result->free_indices.size(), 2);
    ASSERT_EQ(interior_result->factorized_free_hessian.rows(), 2);

    // Reusing the same solver on a corner optimum must not hand that factorization back: the free
    // set is empty, so there is nothing consistent for it to factorize.
    const auto corner_result = solver.solve(hessian, Eigen::Vector2d(-5.0, -5.0), lower, upper);
    ASSERT_TRUE(corner_result.has_value());
    ASSERT_EQ(corner_result->free_indices.size(), 0);
    EXPECT_EQ(corner_result->factorized_free_hessian.rows(), 0);
}

TEST(BoxQPActiveSet, RandomProblemsSatisfyKktAndBeatAProjectedGradientReference)
{
    constexpr int kDim = 4;
    using Mat = Eigen::Matrix<double, kDim, kDim>;
    using Vec = Eigen::Vector<double, kDim>;

    std::mt19937 generator(20260828);
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);
    const auto sample = [&](auto& matrix) {
        for (int i = 0; i < matrix.size(); ++i) matrix.data()[i] = uniform(generator);
    };

    BoxQPActiveSetMethod<kDim> solver;

    for (int draw = 0; draw < 200; ++draw)
    {
        Mat factor;
        sample(factor);
        const Mat hessian = factor.transpose() * factor + Mat::Identity();

        Vec gradient;
        sample(gradient);
        gradient *= 5.0;  // pushes the unconstrained minimizer well outside the box

        Vec lower;
        Vec upper;
        sample(lower);
        sample(upper);
        const Vec box_lower = lower.cwiseMin(upper);
        const Vec box_upper = lower.cwiseMax(upper);

        const auto result = solver.solve(hessian, gradient, box_lower, box_upper);
        ASSERT_TRUE(result.has_value()) << "draw " << draw;

        const Vec& x = result->optimal_vector;
        EXPECT_TRUE((x.array() >= box_lower.array() - 1e-12).all()) << "draw " << draw;
        EXPECT_TRUE((x.array() <= box_upper.array() + 1e-12).all()) << "draw " << draw;
        EXPECT_LT(kkt_residual<kDim>(hessian, gradient, box_lower, box_upper, x), 1e-7)
            << "draw " << draw;

        const Vec reference =
            projected_gradient_reference<kDim>(hessian, gradient, box_lower, box_upper);
        EXPECT_LE(cost<kDim>(hessian, gradient, x),
                  cost<kDim>(hessian, gradient, reference) + 1e-9)
            << "draw " << draw;

        // The active set must be in range, strictly ascending, and sized to match the
        // factorization it indexes into.
        const auto& free_indices = result->free_indices;
        EXPECT_EQ(free_indices.size(), result->factorized_free_hessian.rows()) << "draw " << draw;
        for (int position = 0; position < free_indices.size(); ++position)
        {
            EXPECT_GE(free_indices(position), 0) << "draw " << draw;
            EXPECT_LT(free_indices(position), kDim) << "draw " << draw;
            if (position > 0)
            {
                EXPECT_LT(free_indices(position - 1), free_indices(position)) << "draw " << draw;
            }
        }
    }
}

TEST(BoxQPActiveSet, WarmStartingFromAPreviousSolutionCutsIterations)
{
    // A deliberately hard instance: ill-conditioned, strongly coupled Hessian with a tight,
    // asymmetric box, so the active set only settles after several add/drop rounds. Picked by
    // scanning seeds for the highest cold-start iteration count.
    constexpr int kDim = 10;
    using Mat = Eigen::Matrix<double, kDim, kDim>;
    using Vec = Eigen::Vector<double, kDim>;

    std::mt19937 generator(61);
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);

    Mat factor;
    for (int i = 0; i < factor.size(); ++i) factor.data()[i] = uniform(generator);
    const Mat hessian = factor.transpose() * factor + Mat::Identity() * 0.05;

    Vec gradient;
    for (int i = 0; i < kDim; ++i) gradient(i) = uniform(generator) * 20.0;

    Vec first;
    Vec second;
    for (int i = 0; i < kDim; ++i)
    {
        first(i) = uniform(generator);
        second(i) = uniform(generator);
    }
    const Vec lower = first.cwiseMin(second);
    const Vec upper = first.cwiseMax(second);

    BoxQPActiveSetMethod<kDim> solver;

    const auto cold = solver.solve(hessian, gradient, lower, upper);
    ASSERT_TRUE(cold.has_value());
    const int cold_iterations = solver.iterations();
    // Guards the comparison below: on a one-iteration problem "fewer iterations" proves nothing.
    ASSERT_GE(cold_iterations, 4);
    EXPECT_LT(kkt_residual<kDim>(hessian, gradient, lower, upper, cold->optimal_vector), 1e-7);

    const auto warm = solver.solve(hessian, gradient, lower, upper, cold->optimal_vector);
    ASSERT_TRUE(warm.has_value());

    EXPECT_TRUE(warm->optimal_vector.isApprox(cold->optimal_vector, 1e-10));
    EXPECT_EQ(solver.iterations(), 0);
    // The active set is re-derived from the warm-started point, not carried over.
    ASSERT_EQ(warm->free_indices.size(), cold->free_indices.size());
    EXPECT_TRUE((warm->free_indices == cold->free_indices).all());
}

TEST(BoxQPActiveSet, ProjectionActivatesManyBoundsInOneStep)
{
    // The unconstrained minimizer violates every bound. A ratio-test active-set method would need
    // one iteration per activated bound; the projected step clamps them all at once.
    constexpr int kDim = 8;
    using Mat = Eigen::Matrix<double, kDim, kDim>;
    using Vec = Eigen::Vector<double, kDim>;

    const Mat hessian = Mat::Identity() + Mat::Constant(0.05);
    const Vec gradient = Vec::Constant(-50.0);
    const Vec lower = Vec::Constant(-1.0);
    const Vec upper = Vec::Constant(1.0);

    BoxQPActiveSetMethod<kDim> solver;
    const auto result = solver.solve(hessian, gradient, lower, upper);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->optimal_vector.isApprox(upper, 1e-12));
    EXPECT_EQ(result->free_indices.size(), 0);
    EXPECT_LE(solver.iterations(), 2)
        << "all " << kDim << " bounds should activate in one step";
}

TEST(BoxQPActiveSet, ReturnsNulloptWhenTheFreeHessianIsIndefinite)
{
    Eigen::Matrix2d hessian;
    hessian << 1.0, 0.0, 0.0, -1.0;
    const Eigen::Vector2d gradient(1.0, 1.0);
    const Eigen::Vector2d lower = Eigen::Vector2d::Constant(-100.0);
    const Eigen::Vector2d upper = Eigen::Vector2d::Constant(100.0);

    BoxQPActiveSetMethod<2> solver;
    EXPECT_FALSE(solver.solve(hessian, gradient, lower, upper).has_value());
}

TEST(BoxQPActiveSet, HandlesInfiniteBounds)
{
    Eigen::Matrix2d hessian;
    hessian << 2.0, 0.5, 0.5, 1.0;
    const Eigen::Vector2d gradient(-1.0, -4.0);

    BoxQPActiveSetMethod<2> solver;

    // Fully unbounded: the cold-start midpoint is not finite and must fall back to the origin.
    const auto unbounded =
        solver.solve(hessian, gradient, Eigen::Vector2d::Constant(-kInfinity),
                     Eigen::Vector2d::Constant(kInfinity));
    ASSERT_TRUE(unbounded.has_value());
    EXPECT_TRUE(unbounded->optimal_vector.isApprox(hessian.ldlt().solve(-gradient), 1e-10));
    EXPECT_TRUE((unbounded->free_indices == Eigen::Array2i(0, 1)).all());

    // One-sided: only the second component's upper bound can bind.
    const auto one_sided = solver.solve(hessian, gradient, Eigen::Vector2d(-kInfinity, -kInfinity),
                                        Eigen::Vector2d(kInfinity, 2.0));
    ASSERT_TRUE(one_sided.has_value());
    EXPECT_NEAR(one_sided->optimal_vector(0), 0.0, 1e-10);
    EXPECT_NEAR(one_sided->optimal_vector(1), 2.0, 1e-10);
}

TEST(BoxQPActiveSet, SolvesInSinglePrecision)
{
    Eigen::Matrix3f hessian = Eigen::Matrix3f::Identity() * 2.0F;
    const Eigen::Vector3f gradient(-1.0F, -8.0F, 3.0F);
    const Eigen::Vector3f lower = Eigen::Vector3f::Constant(-2.0F);
    const Eigen::Vector3f upper = Eigen::Vector3f::Constant(2.0F);

    BoxQPConfig<float> config;
    config.convergence.grad_tol = 1e-5F;
    config.convergence.rel_cost_improvement_tol = 1e-6F;

    BoxQPActiveSetMethod<3, float> solver(config);
    const auto result = solver.solve(hessian, gradient, lower, upper);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->optimal_vector(0), 0.5F, 1e-5F);   // interior
    EXPECT_NEAR(result->optimal_vector(1), 2.0F, 1e-5F);   // clamped at the upper bound
    EXPECT_NEAR(result->optimal_vector(2), -1.5F, 1e-5F);  // interior
}

}  // namespace
