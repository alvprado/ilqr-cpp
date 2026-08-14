#pragma once

// Third-party library autodiff
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <utility>

#include "ilqr/math/concepts.hpp"

namespace ilqr
{

/// @brief Automatic Differentiation policy for dynamic jacobians
struct AutoDiff
{
    template <typename DiscreteStep_T, math::ColumnVector StateVec_T,
              math::ColumnVector ControlVec_T>
        requires math::StateTransition<DiscreteStep_T, StateVec_T, ControlVec_T>
    auto operator()(const DiscreteStep_T& f, const StateVec_T& x, const ControlVec_T& u) const
    {
        using namespace autodiff;

        using Scalar = typename StateVec_T::Scalar;
        constexpr auto nx = StateVec_T::RowsAtCompileTime;
        constexpr auto nu = ControlVec_T::RowsAtCompileTime;

        Eigen::Matrix<real, nx, 1> x_ad = x.template cast<real>();
        Eigen::Matrix<real, nu, 1> u_ad = u.template cast<real>();
        Eigen::Matrix<real, nx, 1> F;

        Eigen::Matrix<Scalar, nx, nx> const A = jacobian(f, wrt(x_ad), at(x_ad, u_ad), F);
        Eigen::Matrix<Scalar, nx, nu> const B = jacobian(f, wrt(u_ad), at(x_ad, u_ad), F);

        return std::pair{A, B};
    }
};

}  // namespace ilqr
