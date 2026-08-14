#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>  // std::plus
#include <numeric>     // std::transform_reduce
#include <ranges>      // std::views::iota
#include <utility>     // std::move

#include "solver.hpp"

namespace ilqr
{

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
ILQRSolver<Dynamics_T, CostFunction_T>::ILQRSolver(Dynamics_T dynamics,
                                                   CostFunction_T cost_function,
                                                   const SolverConfig<Scalar>& config)
    : dynamics_(std::move(dynamics)), cost_function_(std::move(cost_function)), config_(config)
{
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::solve(const SolveRequest& solve_request) const
    -> Result
{
    return solve_impl(solve_request, nullptr);
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::solve(const SolveRequest& solve_request,
                                                   SolveDiagnostics& solve_diagnostics) const
    -> Result
{
    return solve_impl(solve_request, &solve_diagnostics);
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::solve_impl(const SolveRequest& solve_request,
                                                        SolveDiagnostics* solve_diagnostics) const
    -> Result
{
    // Clear the diagnostics
    if (solve_diagnostics) solve_diagnostics->iterations.clear();

    // Check for an invalid solve request
    if (solve_request.horizon() <= 0)
    {
        return Result{
            .trajectory = {}, .feedback_gains = {}, .status = SolverStatus::InvalidProblem};
    }

    // Create initial nominal trajectory
    auto nominal_trajectory =
        rollout_trajectory(solve_request.initial_state(), solve_request.initial_controls());
    // Evaluate initial cost
    auto nominal_trajectory_cost = evaluate_trajectory_cost(nominal_trajectory);
    // Instantiate backward pass result
    BackwardPassResult backward_pass_result;
    // Initialize regularization for backward pass
    Scalar reg = config_.regularization.init;

    for (int it = 0; it < config_.max_iterations; ++it)
    {
        // Set-up per-iteration diagnostics recorder
        details::IterationRecorder recorder{solve_diagnostics};
        auto& record = recorder.record;
        record.cost = nominal_trajectory_cost;

        // Backward pass with regularization
        auto maybe_backward_pass_result =
            backward_pass_with_regularization(nominal_trajectory, reg);
        record.reg = reg;

        if (!maybe_backward_pass_result.has_value())
        {
            return Result{
                .trajectory = {}, .feedback_gains = {}, .status = SolverStatus::MaxRegularization};
        }
        backward_pass_result = *std::move(maybe_backward_pass_result);

        // Evaluate gradient norm for termination condition
        Scalar const gradient_norm =
            normalized_gradient_norm(nominal_trajectory.controls(), backward_pass_result.k_ff);

        record.grad_norm = gradient_norm;
        if ((gradient_norm < config_.convergence.grad_tol) && (reg < Scalar(1e-4)))
        {
            return Result{.trajectory = std::move(nominal_trajectory),
                          .feedback_gains = std::move(backward_pass_result.K),
                          .status = SolverStatus::Converged};
        }

        // Forward pass with line search
        auto maybe_forward_pass_result = forward_pass_with_line_search(
            nominal_trajectory, nominal_trajectory_cost, backward_pass_result);

        // If the forward pass did not deliver a suitable trajectory, increase regularization and
        // continue with next iteration
        if (!maybe_forward_pass_result)
        {
            reg = std::max(config_.regularization.min, reg * config_.regularization.factor);
            if (reg >= config_.regularization.max)
            {
                return Result{.trajectory = std::move(nominal_trajectory),
                              .feedback_gains = std::move(backward_pass_result.K),
                              .status = SolverStatus::MaxRegularization};
            }
            continue;
        }
        auto forward_pass_result = *std::move(maybe_forward_pass_result);
        std::swap(nominal_trajectory, forward_pass_result.candidate_trajectory);

        // Record the accepted step
        record.step_size = forward_pass_result.line_search_step_size;
        record.cost = forward_pass_result.candidate_trajectory_cost;
        record.accepted_forward_pass = true;

        // Evaluate relative cost reduction for termination condition
        const Scalar actual_relative_cost_reduction =
            (nominal_trajectory_cost - forward_pass_result.candidate_trajectory_cost) /
            (std::abs(nominal_trajectory_cost) + Scalar(1e-6));

        // Swap costs
        nominal_trajectory_cost = forward_pass_result.candidate_trajectory_cost;

        if (actual_relative_cost_reduction < config_.convergence.cost_tol)
        {
            return Result{.trajectory = std::move(nominal_trajectory),
                          .feedback_gains = std::move(backward_pass_result.K),
                          .status = SolverStatus::Converged};
        }

        // If convergence has not been reach despite an accepted trajectory from the forward pass,
        // relax regularization and iterate again
        reg = reg / config_.regularization.factor;
        reg = (reg < config_.regularization.min) ? Scalar(0) : reg;
    }

    return Result{.trajectory = std::move(nominal_trajectory),
                  .feedback_gains = std::move(backward_pass_result.K),
                  .status = SolverStatus::MaxIterations};
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::rollout_trajectory(
    const StateVec& initial_state, const AlignedVec<ControlVec>& controls) const -> Trajectory
{
    const int horizon_length = static_cast<int>(controls.size());

    Trajectory trajectory(horizon_length);
    trajectory.state(0) = initial_state;

    for (int k = 0; k < horizon_length; ++k)
    {
        trajectory.control(k) = controls.at(k);
        trajectory.state(k + 1) = dynamics_.step(trajectory.state(k), trajectory.control(k));
    }

    return trajectory;
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::evaluate_trajectory_cost(
    const Trajectory& trajectory) const -> Scalar
{
    const int N = trajectory.horizon();
    assert(N > 0 && "Trajectory should not be empty to compute its cost");

    auto const ks = std::views::iota(0, N);
    auto const final_cost = cost_function_.evaluate_final(trajectory.state(N));

    return std::transform_reduce(
        ks.begin(), ks.end(), final_cost, std::plus<>{},
        [&](int k)
        { return cost_function_.evaluate(trajectory.state(k), trajectory.control(k), k); });
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::backward_pass(const Trajectory& trajectory,
                                                           Scalar regularization) const
    -> std::optional<BackwardPassResult>
{
    const int N = trajectory.horizon();
    assert(N > 0 && "Trajectory should not be empty to compute the backward pass");

    auto const final_cost_expansion = cost_function_.quadratize_final(trajectory.state(N));
    auto V_x = final_cost_expansion.lf_x;
    auto V_xx = final_cost_expansion.lf_xx;

    auto k_ff = AlignedVec<ControlVec>(N);
    auto K = AlignedVec<ControlStateMat>(N);
    Scalar dV_lin{Scalar(0)};
    Scalar dV_quad{Scalar(0)};

    for (int k = (N - 1); k >= 0; --k)
    {
        // Symmetrize V_xx
        V_xx = Scalar(0.5) * (V_xx + V_xx.transpose()).eval();

        // Get state and control
        auto const& x = trajectory.state(k);
        auto const& u = trajectory.control(k);

        // Get linear dynamics
        auto const linear_dynamics = dynamics_.linearize(x, u);
        auto const& A = linear_dynamics.A;
        auto const& B = linear_dynamics.B;

        // Get quadratic running cost
        auto const quadratic_running_cost = cost_function_.quadratize(x, u, k);
        auto const& l_x = quadratic_running_cost.l_x;
        auto const& l_u = quadratic_running_cost.l_u;
        auto const& l_xx = quadratic_running_cost.l_xx;
        auto const& l_uu = quadratic_running_cost.l_uu;
        auto const& l_ux = quadratic_running_cost.l_ux;

        // Build Q-value function derivatives
        const ControlStateMat B_T_V_xx = B.transpose() * V_xx;

        const StateVec Q_x = l_x + A.transpose() * V_x;
        const ControlVec Q_u = l_u + B.transpose() * V_x;
        const StateMat Q_xx = l_xx + A.transpose() * V_xx * A;
        const ControlStateMat Q_ux = l_ux + B_T_V_xx * A;
        const ControlMat Q_uu = l_uu + B_T_V_xx * B;

        // Regularized terms for ill-conditioned problems
        const ControlStateMat B_T_Vxx_reg =
            B.transpose() * (V_xx + regularization * StateMat::Identity());
        const ControlStateMat Q_ux_reg = l_ux + B_T_Vxx_reg * A;
        const ControlMat Q_uu_reg = l_uu + B_T_Vxx_reg * B;

        // Q_uu Cholesky decomposition on regularized Q-terms
        Eigen::LLT<ControlMat> llt(Q_uu_reg);
        if (llt.info() != Eigen::Success)
        {
            return std::nullopt;
        }

        // Compute control policy
        k_ff[k] = -llt.solve(Q_u);
        K[k] = -llt.solve(Q_ux_reg);

        // Accumulate expected cost reduction terms
        dV_lin += k_ff[k].dot(Q_u);
        dV_quad += Scalar(0.5) * k_ff[k].dot(Q_uu * k_ff[k]);

        // Transpose
        auto const K_T = K[k].transpose();
        auto const Q_xu = Q_ux.transpose();

        // Update Value function derivatives with the original/un-regularized expressions
        V_x = Q_x + K_T * Q_uu * k_ff[k] + K_T * Q_u + Q_xu * k_ff[k];
        V_xx = Q_xx + K_T * Q_uu * K[k] + K_T * Q_ux + Q_xu * K[k];
    }

    return BackwardPassResult{
        .k_ff = std::move(k_ff), .K = std::move(K), .dV_lin = dV_lin, .dV_quad = dV_quad};
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::backward_pass_with_regularization(
    const Trajectory& trajectory, Scalar& regularization) const -> std::optional<BackwardPassResult>
{
    for (int i = 0; i < config_.regularization.max_iterations; ++i)
    {
        auto maybe_backward_pass_result = backward_pass(trajectory, regularization);
        // Successful backward pass iteration
        if (maybe_backward_pass_result.has_value())
        {
            return maybe_backward_pass_result;
        }

        // If no backward pass result, increase regularization and continue
        regularization =
            std::max(config_.regularization.min, regularization * config_.regularization.factor);
        // regularization ceiling reached - problem is ill-conditioned
        if (regularization >= config_.regularization.max)
        {
            return std::nullopt;
        }
    }

    // Max nr. of backward pass iterations reached - problem is ill-conditioned
    return std::nullopt;
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::forward_pass(
    const Trajectory& nominal_trajectory, const BackwardPassResult& backward_pass_result,
    Scalar step_size) const -> Trajectory
{
    int const N = nominal_trajectory.horizon();
    Trajectory updated_trajectory(N);
    updated_trajectory.state(0) = nominal_trajectory.state(0);

    for (int k = 0; k < N; ++k)
    {
        updated_trajectory.control(k) =
            nominal_trajectory.control(k) + step_size * backward_pass_result.k_ff[k] +
            backward_pass_result.K[k] * (updated_trajectory.state(k) - nominal_trajectory.state(k));

        updated_trajectory.state(k + 1) =
            dynamics_.step(updated_trajectory.state(k), updated_trajectory.control(k));
    }

    return updated_trajectory;
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::forward_pass_with_line_search(
    const Trajectory& nominal_trajectory, Scalar nominal_trajectory_cost,
    const BackwardPassResult& backward_pass_result) const -> std::optional<ForwardPassResult>
{
    Scalar line_search_step_size = Scalar(1.0);
    for (int i = 0; i < config_.line_search.steps; ++i)
    {
        // Rollout trajectory
        auto candidate_trajectory =
            forward_pass(nominal_trajectory, backward_pass_result, line_search_step_size);

        // Evaluate candidate trajectory for acceptance criteria
        Scalar candidate_trajectory_cost = evaluate_trajectory_cost(candidate_trajectory);

        Scalar const actual_cost_reduction = nominal_trajectory_cost - candidate_trajectory_cost;
        Scalar const predicted_cost_reduction =
            -(line_search_step_size * backward_pass_result.dV_lin +
              line_search_step_size * line_search_step_size * backward_pass_result.dV_quad);

        // Line search termination criteria - Armijo actual sufficient cost reduction w.r.t.
        // expected cost reduction. If the predicted cost reduction is negative, the model is not to
        // be trusted and an actual cost reduction is considered
        bool const rel_cost_reduction_accepted =
            (predicted_cost_reduction > Scalar(0))
                ? (actual_cost_reduction >=
                   config_.line_search.min_accept_ratio * predicted_cost_reduction)
                : (actual_cost_reduction > Scalar(0));

        if (rel_cost_reduction_accepted)
        {
            return ForwardPassResult{.candidate_trajectory = std::move(candidate_trajectory),
                                     .candidate_trajectory_cost = candidate_trajectory_cost,
                                     .line_search_step_size = line_search_step_size};
        }

        // Compute line search step size αᵢ = ratio^i
        line_search_step_size *= config_.line_search.ratio;
    }

    // Forward pass did not produce a candidate trajectory with sufficient cost reduction
    return std::nullopt;
}

template <Dynamics Dynamics_T, CostFunction CostFunction_T>
auto ILQRSolver<Dynamics_T, CostFunction_T>::normalized_gradient_norm(
    std::span<const ControlVec> controls, std::span<const ControlVec> feedforward_terms) const
    -> Scalar
{
    return std::transform_reduce(
               controls.begin(), controls.end(), feedforward_terms.begin(), Scalar(0.0),
               std::plus<>{},
               [](const ControlVec& u, const ControlVec& k_ff) {
                   return (k_ff.cwiseAbs().array() / (u.cwiseAbs().array() + Scalar(1.0)))
                       .maxCoeff();
               }) /
           Scalar(controls.size());
}

}  // namespace ilqr