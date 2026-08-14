#pragma once

#include <cmath>
#include <ilqr/ilqr.hpp>

namespace cartpole
{

using Scalar = double;

/// @brief Problem dimensions: state = [x, theta, x_dot, theta_dot], control = [force].
using Dims = ilqr::Dims<4, 1, Scalar>;
using StateVec = Dims::StateVec;
using ControlVec = Dims::ControlVec;
using StateMat = Dims::StateMat;
using ControlMat = Dims::ControlMat;

/// @brief Physical parameters of the cart-pole.
struct CartPoleParameters
{
    Scalar cart_mass{1.0};    ///< Cart mass [kg].
    Scalar pole_mass{0.1};    ///< Pole mass [kg].
    Scalar pole_length{1.0};  ///< Full rod length 2l [m]; the center of gravity sits at l.
};

/// @brief Continuous-time cart-pole model. The pole is a uniform rod of length pole_length (2l),
/// mass pole_mass, with moment of inertia m*(2l)^2/12 about its center. theta is measured from the
/// upright (theta = 0 balanced, theta = pi hanging down).
struct CartPole
{
    CartPoleParameters params{};  ///< Masses, rod length and gravity.

    /// @brief State derivative ẋ = f(x, u) of the cart-pole dynamics.
    /// @tparam Scalar_T Scalar type (double for rollout, autodiff's active type for Jacobians).
    /// @param state Current state [x, theta, x_dot, theta_dot].
    /// @param control Current control [force] applied to the cart.
    /// @return The state derivative [x_dot, theta_dot, x_ddot, theta_ddot].
    template <typename Scalar_T>
    Eigen::Vector<Scalar_T, 4> operator()(const Eigen::Vector<Scalar_T, 4>& state,
                                          const Eigen::Vector<Scalar_T, 1>& control) const
    {
        using std::cos;
        using std::sin;

        // Parameter aliases
        const auto& cart_mass = params.cart_mass;
        const auto& pole_mass = params.pole_mass;
        const auto& pole_length = params.pole_length;

        // States
        const Scalar_T theta = state(1);
        const Scalar_T x_dot = state(2);
        const Scalar_T theta_dot = state(3);

        // Control
        const Scalar_T force = control(0);

        // Helper variables
        const Scalar_T sin_t = sin(theta);
        const Scalar_T cos_t = cos(theta);
        const Scalar_T den = 4 * (pole_mass + cart_mass) - 3 * pole_mass * cos_t * cos_t;
        const Scalar_T half_pole_length = pole_length / 2.0;
        const Scalar gravity = 9.81;

        // Accelerations
        const Scalar_T x_ddot = (4 * force + 3 * pole_mass * gravity * sin_t * cos_t -
                                 4 * pole_mass * half_pole_length * theta_dot * theta_dot * sin_t) /
                                den;
        const Scalar_T theta_ddot =
            (3 * (cart_mass + pole_mass) * gravity * sin_t +
             3 * cos_t * (force - pole_mass * half_pole_length * sin_t * theta_dot * theta_dot)) /
            (half_pole_length * den);

        // State
        return Eigen::Vector<Scalar_T, 4>{x_dot, theta_dot, x_ddot, theta_ddot};
    }
};

static_assert(ilqr::DynamicModel<CartPole, Dims>,
              "Cartpole model must satisfy DynamicModel requirements.");

}  // namespace cartpole
