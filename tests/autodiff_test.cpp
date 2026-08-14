// Unit tests for autodiff third-party library

#include <gtest/gtest.h>

#include "ilqr/dynamics/autodiff_policy.hpp"
#include "ilqr/ilqr.hpp"

namespace
{

// Scalar-generic pendulum: theta_ddot = -(g/l) sin(theta) + u. Its operator() is templated on the
// scalar only, taking fixed-size Eigen::Vector<Scalar, N>, so it evaluates in plain double (finite
// differences / rollout) AND in autodiff's active scalar type — which is what the AutoDiff policy
// needs (the policy hands the model fixed-size active-type vectors).
struct GenericPendulum
{
    using Dims = ilqr::Dims<2, 1>;
    double gravity_over_length = 9.81 / 0.5;

    template <typename Scalar>
    Eigen::Vector<Scalar, 2> operator()(const Eigen::Vector<Scalar, 2>& x,
                                        const Eigen::Vector<Scalar, 1>& u) const
    {
        using std::sin;  // ADL picks autodiff::sin for the active type, std::sin for double
        Eigen::Vector<Scalar, 2> dx;
        dx(0) = x(1);
        dx(1) = -gravity_over_length * sin(x(0)) + u(0);
        return dx;
    }
};

// AutoDiff must produce the same discrete Jacobians as finite differences (to central-difference
// accuracy) for the same model + integrator.
TEST(AutoDiff, MatchesFiniteDifferenceJacobians)
{
    using Dims = ilqr::Dims<2, 1>;
    const double dt = 0.02;

    // Same model, same integrator (RK4); only the differentiation policy differs.
    auto dyn_fd = ilqr::discretize<Dims>(GenericPendulum{}, dt, ilqr::math::RK4Step{},
                                         ilqr::FiniteDifference{});
    auto dyn_ad =
        ilqr::discretize<Dims>(GenericPendulum{}, dt, ilqr::math::RK4Step{}, ilqr::AutoDiff{});

    Dims::StateVec x;
    x << 0.7, -0.4;  // away from equilibrium so the Jacobians are nontrivial
    Dims::ControlVec u;
    u << 0.3;

    const auto fd = dyn_fd.linearize(x, u);
    const auto ad = dyn_ad.linearize(x, u);

    EXPECT_TRUE(ad.A.isApprox(fd.A, 1e-6)) << "A_ad=\n" << ad.A << "\nA_fd=\n" << fd.A;
    EXPECT_TRUE(ad.B.isApprox(fd.B, 1e-6)) << "B_ad=\n" << ad.B << "\nB_fd=\n" << fd.B;
}

}  // namespace
