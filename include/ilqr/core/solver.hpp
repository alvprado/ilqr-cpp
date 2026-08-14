#pragma once
#include <optional>

#include "ilqr/core/config.hpp"
#include "ilqr/cost/cost_concepts.hpp"
#include "ilqr/dynamics/dynamics_concepts.hpp"
#include "ilqr/core/solver_diagnostics.hpp"
#include "ilqr/core/solver_io.hpp"
#include "ilqr/core/trajectory.hpp"
#include "ilqr/core/types.hpp"

namespace ilqr
{

/// @brief Iterative Linear-Quadratic Regulator (iLQR) solver.
/// @details Given a dynamics model and a cost function, repeatedly improves an initial control
/// guess: each iteration linearizes the dynamics and quadratizes the cost about the current
/// nominal trajectory (backward pass) to produce a locally optimal affine control update, then
/// applies it with a backtracking line search (forward pass). Iterates until the feedforward
/// gradient or the relative cost reduction falls below tolerance, the iteration limit is hit, or
/// the problem proves ill-conditioned. Regularization is adapted across iterations to keep the
/// control Hessian positive-definite.
/// @tparam Dynamics_T A model satisfying the Dynamics concept (exposes Dims, step(x, u) and
///         linearize(x, u)).
/// @tparam CostFunction_T A cost satisfying the CostFunction concept (exposes Dims, evaluate,
///         quadratize, evaluate_final and quadratize_final). Must share Dims with Dynamics_T.
template <Dynamics Dynamics_T, CostFunction CostFunction_T>
class ILQRSolver
{
    /// Dimension alias
    using Dims = typename Dynamics_T::Dims;

    /// Check dynamics and costs share the same dimensions
    static_assert(std::is_same_v<Dims, typename CostFunction_T::Dims>,
                  "Dynamics and CostFunction must share the same dimensions");

    /// Aliases
    using Scalar = typename Dims::Scalar;
    using StateVec = typename Dims::StateVec;
    using ControlVec = typename Dims::ControlVec;
    using StateMat = typename Dims::StateMat;
    using ControlMat = typename Dims::ControlMat;
    using ControlStateMat = typename Dims::ControlStateMat;
    using Trajectory = Trajectory<Dims>;
    using Result = Result<Dims>;
    using SolveRequest = SolveRequest<Dims>;
    using SolveDiagnostics = SolveDiagnostics<Scalar>;

    /// @brief The feedforward and feedback gains of a backward pass, plus the expected cost
    ///        reduction the resulting affine policy predicts.
    struct BackwardPassResult
    {
        /// Feedforward terms k_ff[k] = -Q_uu^-1 Q_u, one per timestep
        AlignedVec<ControlVec> k_ff;
        /// Feedback gains K[k] = -Q_uu^-1 Q_ux (du = K[k] dx), one per timestep
        AlignedVec<ControlStateMat> K;
        /// Linear term of the predicted cost change: sum_k k_ff[k]^T Q_u[k]
        Scalar dV_lin{Scalar(0)};
        /// Quadratic term of the predicted cost change: sum_k 1/2 k_ff[k]^T Q_uu[k] k_ff[k]
        Scalar dV_quad{Scalar(0)};
    };

    /// @brief The outcome of an accepted forward pass: the candidate trajectory, its cost and the
    ///        line search step size that produced it.
    struct ForwardPassResult
    {
        /// Candidate trajectory rolled out with the accepted control update
        Trajectory candidate_trajectory;
        /// Total cost of the candidate trajectory
        Scalar candidate_trajectory_cost;
        /// Accepted line search step size alpha applied to the feedforward term
        Scalar line_search_step_size;
    };

public:
    /// @brief Construct the solver from its dynamics, cost function and configuration.
    /// @param[in] dynamics The system dynamics model (taken by value, moved into the solver).
    /// @param[in] cost_function The cost function (taken by value, moved into the solver).
    /// @param[in] config Solver configuration: iteration limit, regularization schedule, line
    ///        search and convergence tolerances.
    ILQRSolver(Dynamics_T dynamics, CostFunction_T cost_function,
               const SolverConfig<Scalar>& config);

    /// @brief Solve an iterative LQR problem given its initial state and initial guess on the
    /// controls
    /// @param[in] solve_request A solver request containing the initial states x0 and the initial
    /// guess of control inputs U_init = [u_0, ..., u_N-1]
    /// @returns A result containing an optimal iLQR trajectory, the optimal control policy and the
    /// solver status
    [[nodiscard]] Result solve(const SolveRequest& solve_request) const;

    /// @brief Solve an iterative LQR problem and record per-iteration diagnostics.
    /// @param[in] solve_request A solver request containing the initial state x0 and the initial
    ///        guess of control inputs U_init = [u_0, ..., u_N-1].
    /// @param[out] solve_diagnostics Populated with one IterationRecord per outer iteration
    ///        (including iterations that exit early without an accepted step). Cleared on entry.
    /// @returns A result containing the optimized trajectory, the feedback policy and the status.
    [[nodiscard]] Result solve(const SolveRequest& solve_request,
                               SolveDiagnostics& solve_diagnostics) const;

private:
    /// @brief Shared implementation behind both solve() overloads.
    /// @param[in] solve_request The problem to solve.
    /// @param[out] solve_diagnostics Non-owning pointer to a diagnostics sink to fill (cleared on
    ///        entry), or nullptr to skip recording (the plain solve() path).
    /// @returns A result containing the optimized trajectory, the feedback policy and the status.
    [[nodiscard]] Result solve_impl(const SolveRequest& solve_request,
                                    SolveDiagnostics* solve_diagnostics) const;

    /// @brief Forward-simulate the dynamics from an initial state under a control sequence.
    /// @param[in] initial_state The initial state x0.
    /// @param[in] controls The control sequence [u_0, ..., u_{N-1}]; its length sets the horizon N.
    /// @returns The resulting trajectory (states x_0..x_N and controls u_0..u_{N-1}).
    [[nodiscard]] Trajectory rollout_trajectory(const StateVec& initial_state,
                                                const AlignedVec<ControlVec>& controls) const;

    /// @brief Evaluate the total cost of a trajectory.
    /// @param[in] trajectory The trajectory to evaluate (must be non-empty).
    /// @returns The sum of the running costs over k = 0..N-1 plus the final cost at x_N.
    [[nodiscard]] Scalar evaluate_trajectory_cost(const Trajectory& trajectory) const;

    /// @brief Run a single backward pass at a fixed regularization.
    /// @param[in] trajectory The nominal trajectory to linearize and quadratize about.
    /// @param[in] regularization The value added to the state value Hessian V_xx when forming the
    ///        control-update terms.
    /// @returns The gains and predicted cost reduction, or std::nullopt if the regularized control
    ///          Hessian Q_uu is not positive-definite (Cholesky failure).
    [[nodiscard]] std::optional<BackwardPassResult> backward_pass(const Trajectory& trajectory,
                                                                  Scalar regularization) const;

    /// @brief Run the backward pass, increasing regularization until it succeeds or a ceiling is
    ///        reached.
    /// @param[in] trajectory The nominal trajectory to linearize and quadratize about.
    /// @param[in,out] regularization On entry the starting regularization; on Cholesky failure it
    ///        is grown by the configured factor and retried, and on return it holds the value
    ///        actually used.
    /// @returns The backward pass result, or std::nullopt if the regularization ceiling (or the
    ///          retry limit) is reached without a positive-definite Q_uu.
    [[nodiscard]] std::optional<BackwardPassResult> backward_pass_with_regularization(
        const Trajectory& trajectory, Scalar& regularization) const;

    /// @brief Roll out a single candidate trajectory for a given line search step size.
    /// @param[in] nominal_trajectory The current nominal trajectory to update from.
    /// @param[in] backward_pass_result The gains (k_ff, K) computed by the backward pass.
    /// @param[in] step_size The line search step size alpha scaling the feedforward term:
    ///        u_new[k] = u[k] + alpha k_ff[k] + K[k] (x_new[k] - x[k]).
    /// @returns The candidate trajectory obtained by rolling out the updated controls.
    [[nodiscard]] Trajectory forward_pass(const Trajectory& nominal_trajectory,
                                          const BackwardPassResult& backward_pass_result,
                                          Scalar step_size) const;

    /// @brief Backtracking line search: try decreasing step sizes until one meets the Armijo
    ///        sufficient-decrease criterion.
    /// @param[in] nominal_trajectory The current nominal trajectory to update from.
    /// @param[in] nominal_trajectory_cost The cost of the nominal trajectory, used as the baseline
    ///        in the acceptance test.
    /// @param[in] backward_pass_result The gains and expected cost reduction from the backward pass.
    /// @returns The accepted candidate (trajectory, cost and step size), or std::nullopt if no
    ///          step size in the configured schedule was accepted.
    [[nodiscard]] std::optional<ForwardPassResult> forward_pass_with_line_search(
        const Trajectory& nominal_trajectory, Scalar nominal_trajectory_cost,
        const BackwardPassResult& backward_pass_result) const;

    /// @brief Compute the normalized feedforward-gradient norm used as a convergence measure.
    /// @param[in] controls The nominal control sequence u_0..u_{N-1}.
    /// @param[in] feedforward_terms The feedforward gains k_ff from the backward pass.
    /// @returns The mean over timesteps of the per-timestep maximum of
    ///          |k_ff[k][j]| / (|u[k][j]| + 1); it tends to zero as the trajectory approaches a
    ///          stationary point.
    [[nodiscard]] Scalar normalized_gradient_norm(
        std::span<const ControlVec> controls, std::span<const ControlVec> feedforward_terms) const;

    /// System dynamics
    Dynamics_T dynamics_;

    /// Cost function
    CostFunction_T cost_function_;

    /// Solver config
    SolverConfig<Scalar> config_;
};
}  // namespace ilqr

#include "solver.inl"
