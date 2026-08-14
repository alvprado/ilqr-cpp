#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <Eigen/Dense>
#include <limits>
#include <type_traits>

#include "ilqr/math/concepts.hpp"

namespace ilqr::math
{

/// @brief A callable f(x) returning an Eigen column vector.
/// @tparam Fn_T A callable invoked as f(x).
/// @tparam Vec_T The input (column) vector type.
template <typename Fn_T, typename Vec_T>
concept VectorValuedFunction = requires(const Fn_T& f, const Vec_T& x) {
                                   {
                                       f(x)
                                   } -> ColumnVector;
                               };

/// @brief Central-difference approximation of the Jacobian df/dx evaluated at a point x.
/// @details Evaluates f at x ± h along each input component, with the step h scaled to the
/// component's magnitude. Second-order accurate.
/// @tparam Fn_T A vector-valued callable f(x) (VectorValuedFunction<Fn_T, Vec_T>).
/// @tparam Vec_T The input column-vector type.
/// @param[in] f The function to differentiate.
/// @param[in] x The point at which the Jacobian is evaluated.
/// @returns The Jacobian, with as many rows as f(x) and as many columns as x.
template <typename Fn_T, ColumnVector Vec_T>
    requires VectorValuedFunction<Fn_T, Vec_T>
auto finite_difference_jacobian(const Fn_T& f, const Vec_T& x)
    -> Eigen::Matrix<typename Vec_T::Scalar, std::decay_t<decltype(f(x))>::RowsAtCompileTime,
                     Vec_T::RowsAtCompileTime>
{
    using Scalar = typename Vec_T::Scalar;
    using OutVec = std::decay_t<decltype(f(x))>;

    const Scalar eps = std::cbrt(std::numeric_limits<Scalar>::epsilon());
    Eigen::Matrix<Scalar, OutVec::RowsAtCompileTime, Vec_T::RowsAtCompileTime> jacobian;

    Vec_T x_pert = x;

    for (Eigen::Index i = 0; i < x.size(); i++)
    {
        const Scalar h = eps * std::max(Scalar(1), std::abs(x(i)));
        x_pert(i) = x(i) + h;
        const OutVec f_plus = f(x_pert);
        x_pert(i) = x(i) - h;
        const OutVec f_minus = f(x_pert);
        jacobian.col(i) = (f_plus - f_minus) / (Scalar(2) * h);
        x_pert(i) = x(i);
    }

    return jacobian;
}

}  // namespace ilqr::math