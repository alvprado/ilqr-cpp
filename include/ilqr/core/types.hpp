#pragma once

#include <concepts>
#include <Eigen/Dense>
#include <vector>

namespace ilqr
{

/// @brief Dimension struct containing the number of state and controls, as well as relevant
/// matrixes dimensions as Eigen aliases
/// @tparam state_dim_v Number of states
/// @tparam control_dim_v Number of control inputs
/// @tparam Scalar_T Floating-point scalar
template <int state_dim_v, int control_dim_v, std::floating_point Scalar_T = double>
struct Dims
{
    /// State and control dimensions
    static constexpr int state_dim = state_dim_v;
    static constexpr int control_dim = control_dim_v;

    /// Scalar alias
    using Scalar = Scalar_T;

    /// State and control vector eigen type aliases
    using StateVec = Eigen::Matrix<Scalar, state_dim_v, 1>;
    using ControlVec = Eigen::Matrix<Scalar, control_dim_v, 1>;

    /// State matrix type alias - E.g. dynamics state jacobian A or state cost hessian l_xx
    using StateMat = Eigen::Matrix<Scalar, state_dim_v, state_dim_v>;

    /// Control matrix type alias - E.g. control cost hessian l_uu
    using ControlMat = Eigen::Matrix<Scalar, control_dim_v, control_dim_v>;

    /// State-Control matrix type alias - E.g. dynamics control jacobian B
    using StateControlMat = Eigen::Matrix<Scalar, state_dim_v, control_dim_v>;

    /// Control-State matrix type alias - E.g. control-state cost hessian l_ux or feedback gain K
    using ControlStateMat = Eigen::Matrix<Scalar, control_dim_v, state_dim_v>;  // l_ux, gain K
};

/// @brief Vector with an Eigen aligned allocator for correct storage alignment
template <class T>
using AlignedVec = std::vector<T, Eigen::aligned_allocator<T>>;

}  // namespace ilqr
