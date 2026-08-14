#pragma once

#include <concepts>
#include <ilqr/ilqr.hpp>

namespace quadrotor
{

using Scalar = double;

/// @brief Problem dimensions: state = [x, y, theta, x_dot, y_dot, theta_dot], control = [F1, F2].
using Dims = ilqr::Dims<6, 2, Scalar>;
using StateVec = Dims::StateVec;
using ControlVec = Dims::ControlVec;
using StateMat = Dims::StateMat;
using ControlMat = Dims::ControlMat;

/// @brief Physical parameters of the planar quadrotor
struct QuadrotorParameters
{
    Scalar m_body{5.0};   ///< Central body mass [kg].
    Scalar r_body{0.5};   ///< Body radius [m] (body modelled as a disk for its inertia).
    Scalar m_rotor{0.5};  ///< Mass of each rotor [kg].
    Scalar l{1.0};        ///< Arm half-length: rotor offset from the center [m].
};

/// @brief Continuous dynamic model of a planar quadrotor.
/// @note y-up, theta-clockwise-positive convention: the body thrust axis is (sin theta, cos theta),
/// so gravity enters y_ddot as -g and level hover needs F1 + F2 = mass * g.
struct Quadrotor
{
    QuadrotorParameters params{};  ///< Mass and geometry.

    /// @brief State derivative ẋ = f(x, u) of the planar quadrotor.
    /// @tparam Scalar_T Scalar type (double for rollout, autodiff's active type for Jacobians).
    /// @param states Current state [x, y, theta, x_dot, y_dot, theta_dot].
    /// @param controls Current control [F1, F2] (right/left rotor thrusts).
    /// @return The state derivative [x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot].
    template <typename Scalar_T>
    Eigen::Vector<Scalar_T, 6> operator()(const Eigen::Vector<Scalar_T, 6>& states,
                                          const Eigen::Vector<Scalar_T, 2>& controls) const
    {
        using std::cos;
        using std::sin;

        // States
        const Scalar_T x = states(0);
        const Scalar_T y = states(1);
        const Scalar_T theta = states(2);
        const Scalar_T x_dot = states(3);
        const Scalar_T y_dot = states(4);
        const Scalar_T theta_dot = states(5);

        // Controls
        const Scalar_T force1 = controls(0);
        const Scalar_T force2 = controls(1);

        // Helper variables
        const Scalar mass = params.m_body + 2 * params.m_rotor;
        const Scalar I = 2 * params.m_rotor * params.l * params.l +
                         0.5 * params.m_body * params.r_body * params.r_body;
        const Scalar g = 9.81;

        // Accelerations
        const Scalar_T x_ddot = (force1 + force2) * sin(theta) / mass;
        const Scalar_T y_ddot = ((force1 + force2) * cos(theta) / mass) - g;
        const Scalar_T theta_ddot = (force2 - force1) * params.l / I;

        // State derivative
        return Eigen::Vector<Scalar_T, 6>{x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot};
    }
};

static_assert(ilqr::DynamicModel<Quadrotor, Dims>,
              "Quadrotor model must satisfy DynamicModel requirements.");
}  // namespace quadrotor