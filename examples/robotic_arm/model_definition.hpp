#pragma once

#include <cmath>
#include <ilqr/ilqr.hpp>

namespace robotic_arm
{

using Scalar = double;

/// @brief Problem dimensions: state = [q1, q2, q1_dot, q2_dot], control = [tau1, tau2].
using Dims = ilqr::Dims<4, 2, Scalar>;
using StateVec = Dims::StateVec;
using ControlVec = Dims::ControlVec;
using StateMat = Dims::StateMat;
using ControlMat = Dims::ControlMat;

/// @brief Physical parameters of the 2-link arm
struct RobotParameters
{
    Scalar m1{3.0};  ///< Mass of link 1 [kg].
    Scalar m2{2.5};  ///< Mass of link 2 [kg].
    Scalar l1{1.0};  ///< Length of link 1 [m].
    Scalar l2{0.7};  ///< Length of link 2 [m].
};

/// @brief Continuous dynamic model of a 2-link planar robotic arm.
struct RoboticArm
{
    RobotParameters params{};  ///< Link masses and lengths.

    /// @brief State derivative ẋ = f(x, u); solves M(q) q̈ + C(q, q̇) = τ for joint accelerations.
    /// @tparam Scalar_T Scalar type
    /// @param state Current state [q1, q2, q1_dot, q2_dot].
    /// @param control Current control [tau1, tau2] (joint torques).
    /// @return The state derivative [q1_dot, q2_dot, q1_ddot, q2_ddot].
    template <typename Scalar_T>
    Eigen::Vector<Scalar_T, 4> operator()(const Eigen::Vector<Scalar_T, 4>& state,
                                          const Eigen::Vector<Scalar_T, 2>& control) const
    {
        using std::cos;
        using std::sin;

        // Parameter aliases
        const auto& m1 = params.m1;
        const auto& m2 = params.m2;
        const auto& l1 = params.l1;
        const auto& l2 = params.l2;

        // States
        const Scalar_T q1 = state(0);
        const Scalar_T q2 = state(1);
        const Scalar_T q1_dot = state(2);
        const Scalar_T q2_dot = state(3);

        // Controls
        const Scalar_T torque1 = control(0);
        const Scalar_T torque2 = control(1);

        // Helper variables
        const Scalar_T s2 = sin(q2);
        const Scalar_T c1 = cos(q1);
        const Scalar_T c2 = cos(q2);
        const Scalar_T c12 = cos(q1 + q2);
        const Scalar l1_sq = l1 * l1;
        const Scalar l2_sq = l2 * l2;

        // Mass Matrix entries
        const Scalar m22 = ((m2 * l2_sq) / 3.0);
        const Scalar_T m11 = ((m1 * l1_sq) / 3.0) + (m2 * l1_sq) + m22 + (m2 * l1 * l2) * c2;
        const Scalar_T m12 = m22 + ((m2 * l1 * l2) * c2 / 2.0);

        // Coriolis/centrifugal vector entries
        const Scalar_T h = (m2 * l1 * l2 * s2) / 2.0;
        const Scalar_T cor1 = -h * (2 * q1_dot * q2_dot + q2_dot * q2_dot);
        const Scalar_T cor2 = h * q1_dot * q1_dot;

        // Determinant of mass matrix
        const Scalar_T det_m = m11 * m22 - m12 * m12;

        // RHS vector for solving M q_ddot = rhs
        const Scalar_T rhs1 = torque1 - cor1;
        const Scalar_T rhs2 = torque2 - cor2;

        // Joint accelerations
        const Scalar_T q1_ddot = (m22 * rhs1 - m12 * rhs2) / det_m;
        const Scalar_T q2_ddot = (-m12 * rhs1 + m11 * rhs2) / det_m;

        // State derivative
        return Eigen::Vector<Scalar_T, 4>{q1_dot, q2_dot, q1_ddot, q2_ddot};
    }
};

static_assert(ilqr::DynamicModel<RoboticArm, Dims>,
              "Robotic Arm model must satisfy DynamicModel requirements.");

}  // namespace robotic_arm
