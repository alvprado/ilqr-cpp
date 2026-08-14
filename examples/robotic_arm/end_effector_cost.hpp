#pragma once

// The 2-link arm's bespoke task-space cost.

#include <cmath>
#include <ilqr/ilqr.hpp>
#include <utility>

#include "model_definition.hpp"

namespace robotic_arm
{

/// @brief Config for the end-effector position cost. Composes the shared RobotParameters (for the
/// forward-kinematics arm lengths) with the task-space target and the running/final weights.
struct EndEffectorPositionCostConfig
{
    RobotParameters params{};     ///< Arm lengths used for the forward kinematics.
    Scalar x_position_target{0};  ///< Task-space target x [m].
    Scalar y_position_target{0};  ///< Task-space target y [m].
    Scalar running_weight{0};     ///< Weight applied to the running cost.
    Scalar final_weight{0};       ///< Weight applied to the final cost.
};

/// @brief Penalizes the squared task-space distance of the end effector from a target.
/// @details fk(q) is the forward kinematics of the 2-link arm; the cost is ½ ||fk(q) - target||².
/// The quadratic expansion uses the EXACT gradient and Hessian; that Hessian is generally
/// indefinite, so this cost exercises the solver's regularization (a Gauss-Newton Hessian would
/// instead stay positive semi-definite).
/// @tparam Dims_T The dimensions the cost operates on.
template <typename Dims_T>
class EndEffectorPositionCost
{
public:
    using Dims = Dims_T;
    using Scalar = typename Dims_T::Scalar;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;
    using ControlMat = typename Dims_T::ControlMat;

    /// @brief Build the cost from its config (arm lengths, target, running/final weights).
    EndEffectorPositionCost(EndEffectorPositionCostConfig config) : config_(std::move(config)) {}

    /// @brief Running cost running_weight * ½ ||fk(q) - target||².
    [[nodiscard]] Scalar evaluate(const StateVec& x, const ControlVec&, int) const
    {
        return config_.running_weight * evaluate_impl(x(0), x(1));
    }

    /// @brief Final cost final_weight * ½ ||fk(q) - target||².
    [[nodiscard]] Scalar evaluate_final(const StateVec& x) const
    {
        return config_.final_weight * evaluate_impl(x(0), x(1));
    }

    /// @brief Running-cost quadratic expansion (the final-cost expansion scaled by running_weight).
    [[nodiscard]] ilqr::CostTaylorExpansion<Dims_T> quadratize(const StateVec& x, const ControlVec&,
                                                               int) const
    {
        auto const expansion = quadratize_impl(x(0), x(1), config_.running_weight);
        ilqr::CostTaylorExpansion<Dims_T> result{};
        result.l = expansion.lf;
        result.l_x = expansion.lf_x;
        result.l_xx = expansion.lf_xx;
        return result;
    }

    /// @brief Final-cost quadratic expansion with the exact (indefinite) gradient/Hessian.
    [[nodiscard]] ilqr::FinalCostTaylorExpansion<Dims_T> quadratize_final(const StateVec& x) const
    {
        return quadratize_impl(x(0), x(1), config_.final_weight);
    }

private:
    /// @brief Half squared task-space error at joint angles (q1, q2).
    [[nodiscard]] Scalar evaluate_impl(const Scalar q1, const Scalar q2) const
    {
        auto const error_x = config_.params.l1 * std::cos(q1) +
                             config_.params.l2 * std::cos(q1 + q2) - config_.x_position_target;
        auto const error_y = config_.params.l1 * std::sin(q1) +
                             config_.params.l2 * std::sin(q1 + q2) - config_.y_position_target;
        return Scalar(0.5) * (error_x * error_x + error_y * error_y);
    }

    /// @brief Exact value, gradient and (indefinite) Hessian of the task-space error, scaled by
    /// weight.
    [[nodiscard]] ilqr::FinalCostTaylorExpansion<Dims> quadratize_impl(
        const Scalar q1, const Scalar q2, const Scalar weight = Scalar(1)) const
    {
        // Helper variables
        auto const s1 = std::sin(q1);
        auto const c1 = std::cos(q1);
        auto const s2 = std::sin(q2);
        auto const c2 = std::cos(q2);
        auto const s12 = std::sin(q1 + q2);
        auto const c12 = std::cos(q1 + q2);

        // Helper aliases
        auto const& l1 = config_.params.l1;
        auto const& l2 = config_.params.l2;
        auto const& px = config_.x_position_target;
        auto const& py = config_.y_position_target;

        // Compute gradient and hessian entries
        auto const dl_dq1 = l1 * s1 * px + l2 * s12 * px - l1 * c1 * py - l2 * c12 * py;
        auto const dl_dq2 = -l1 * l2 * s2 + l2 * px * s12 - l2 * py * c12;
        auto const ddl_ddq1 = l1 * px * c1 + l2 * px * c12 + l1 * py * s1 + l2 * py * s12;
        auto const ddl_ddq2 = -l1 * l2 * c2 + l2 * px * c12 + l2 * py * s12;
        auto const ddl_dq1dq2 = l2 * px * c12 + l2 * py * s12;

        ilqr::FinalCostTaylorExpansion<Dims> result;
        result.lf = weight * evaluate_impl(q1, q2);
        result.lf_x(0) = weight * dl_dq1;
        result.lf_x(1) = weight * dl_dq2;
        result.lf_xx(0, 0) = weight * ddl_ddq1;
        result.lf_xx(0, 1) = weight * ddl_dq1dq2;
        result.lf_xx(1, 0) = weight * ddl_dq1dq2;
        result.lf_xx(1, 1) = weight * ddl_ddq2;

        return result;
    }

    EndEffectorPositionCostConfig config_;
};

static_assert(ilqr::CostFunction<EndEffectorPositionCost<Dims>>,
              "EndEffectorPositionCost must satisfy CostFunction requirements.");

}  // namespace robotic_arm
