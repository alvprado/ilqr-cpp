#pragma once

#include <concepts>
#include <Eigen/Dense>
#include <limits>
#include <optional>

namespace ilqr::math
{

/// @brief Config parameters of the box-constrained QP solver
template <std::floating_point Scalar_T = double>
struct BoxQPConfig
{
    /// Relative precision of the Scalar type
    static constexpr Scalar_T eps = std::numeric_limits<Scalar_T>::epsilon();

    /// Maximum number of solver iterations before termination without convergence
    int max_iterations{100};

    /// @brief Convergence specifications
    struct Convergence
    {
        /// Infinity-norm tolerance on the projected gradient
        Scalar_T grad_tol{Scalar_T(1e-8)};
        /// Relative cost improvement below which the iteration is considered stalled
        Scalar_T rel_cost_improvement_tol{Scalar_T(1e-8)};
    } convergence;

    /// @brief Projected backtracking line search along the Newton direction. The step size
    /// starts at 1 and is reduced by alpha_i = ratio^i with i being the step number starting at 0.
    struct LineSearch
    {
        /// Number of trial steps to try
        int steps{100};

        /// Ratio at which the step size decays on a rejected trial step
        Scalar_T ratio{Scalar_T(0.6)};

        /// Minimum fraction of predicted reduction to accept a trial step
        Scalar_T min_accept_ratio{Scalar_T(0.1)};
    } line_search;
};

/// @brief The outcome of a box QP solve: the minimizer plus the active-set information needed to
/// build a constrained feedback policy.
/// @tparam N The problem dimension
/// @tparam Scalar_T Floating-point scalar
template <int N, std::floating_point Scalar_T>
struct BoxQPActiveSetResult
{
    /// The free (unclamped) sub-block of the Hessian, dynamically sized but statically allocated
    using ReducedHessian = Eigen::Matrix<Scalar_T, Eigen::Dynamic, Eigen::Dynamic, 0, N, N>;

    /// Positions of the free components, dynamically sized but statically allocated
    using FreeIndices = Eigen::Array<int, Eigen::Dynamic, 1, 0, N, 1>;

    /// The optimal values of the constraint QP
    Eigen::Vector<Scalar_T, N> optimal_vector;

    /// @brief Ascending positions of the free components
    FreeIndices free_indices;

    /// Cholesky factorization of the Hessian restricted to the free components.
    Eigen::LLT<ReducedHessian> factorized_free_hessian;
};

/// @brief Projected-Newton active-set solver for the box-constrained quadratic program
///        minimize 0.5 x^T H x + g^T x  subject to  lower <= x <= upper.
/// @details Each iteration classifies every component as clamped (sitting on a bound with a
/// gradient that pushes further into it, so its multiplier is KKT-consistent) or free, takes a
/// Newton step on the free subspace only, and projects the trial point back into the box during a
/// backtracking line search.
/// @tparam N The problem dimension
/// @tparam Scalar_T Floating-point scalar
template <int N, std::floating_point Scalar_T = double>
class BoxQPActiveSetMethod
{
    static_assert(N > 0, "Box QP dimension must be positive");

public:
    /// Aliases
    using Scalar = Scalar_T;
    using Hessian = Eigen::Matrix<Scalar_T, N, N>;
    using Gradient = Eigen::Vector<Scalar_T, N>;
    using Vector = Eigen::Vector<Scalar_T, N>;
    using Result = BoxQPActiveSetResult<N, Scalar_T>;

    /// @brief Construct the solver from its configuration.
    /// @param[in] config Box QP configuration containing convergence decision and line search
    /// parameters.
    explicit BoxQPActiveSetMethod(const BoxQPConfig<Scalar_T>& config = {});

    /// @brief Solve the box QP from an explicit starting point (warm start).
    /// @param[in] hessian The Hessian H, which must be positive-definite on the free subspace.
    /// @param[in] gradient The linear term g.
    /// @param[in] lower_bound The componentwise lower bound; -infinity is allowed.
    /// @param[in] upper_bound The componentwise upper bound; +infinity is allowed.
    /// @param[in] initial_guess The starting point, clamped into the box on entry.
    /// @returns The minimizer and its active-set information, or std::nullopt if the solve did not
    /// converge
    [[nodiscard]] std::optional<Result> solve(const Hessian& hessian, const Gradient& gradient,
                                              const Vector& lower_bound, const Vector& upper_bound,
                                              const Vector& initial_guess);

    /// @brief Solve the box QP from a cold start.
    /// @details Starts at the midpoint of the bounds, with components whose midpoint is not finite
    /// (an infinite or doubly-infinite interval) set to zero and then clamped into the box.
    /// @param[in] hessian The Hessian H, which must be positive-definite on the free subspace.
    /// @param[in] gradient The linear term g.
    /// @param[in] lower_bound The componentwise lower bound; -infinity is allowed.
    /// @param[in] upper_bound The componentwise upper bound; +infinity is allowed.
    /// @returns The minimizer and its active-set information, or std::nullopt if the solve did not
    ///          converge
    [[nodiscard]] std::optional<Result> solve(const Hessian& hessian, const Gradient& gradient,
                                              const Vector& lower_bound, const Vector& upper_bound);

    /// @brief Number of projected-Newton iterations the most recent solve() performed.
    /// @details Diagnostic only, and deliberately not part of Result
    /// @returns The iteration count, or zero if the solve converged at its starting point.
    [[nodiscard]] int iterations() const { return iterations_; }

private:
    /// The free sub-block of the Hessian
    using ReducedHessian = typename Result::ReducedHessian;

    /// Positions of the free components
    using FreeIndices = typename Result::FreeIndices;

    /// The gradient restricted to the free components
    using ReducedGradient = Eigen::Matrix<Scalar_T, Eigen::Dynamic, 1, 0, N, 1>;

    /// @brief References to the problem data, which is fixed for the duration of one solve.
    /// @details Bundled because these four always travel together.
    struct Problem
    {
        const Hessian& hessian;
        const Gradient& gradient;
        const Vector& lower_bound;
        const Vector& upper_bound;
    };

    /// @brief Evaluate the quadratic cost 0.5 x^T H x + g^T x.
    /// @param[in] problem The problem data.
    /// @param[in] x The point at which to evaluate.
    /// @returns The cost at x.
    [[nodiscard]] static Scalar evaluate_cost(const Problem& problem, const Vector& x);

    /// @brief Project a point into the box.
    /// @param[in] problem The problem data.
    /// @param[in] x The point to project.
    /// @returns The componentwise clamp of x into [lower_bound, upper_bound].
    [[nodiscard]] static Vector clamp_into_box(const Problem& problem, const Vector& x);

    /// @brief Classify every component as free or clamped, and collect the free ones.
    /// @details Component i is clamped when it sits on a bound and the gradient pushes further into
    /// it: (x_i <= lower_i and grad_i > 0) or (x_i >= upper_i and grad_i < 0). These are exactly
    /// the bounds whose KKT multiplier is strictly positive. The indices come out ascending, which
    /// is the ordering every reduced object is expressed in.
    /// @param[in] problem The problem data.
    /// @param[in] x The current iterate.
    /// @param[in] current_gradient The gradient H x + g at the current iterate.
    /// @returns The positions of the free components, empty when every component is clamped.
    [[nodiscard]] static FreeIndices compute_free_indices(const Problem& problem, const Vector& x,
                                                          const Gradient& current_gradient);

    /// @brief Scatter the free-subspace Newton step into a full-size direction.
    /// @details Solves H_ff d_free = -grad_free with the cached factorization and leaves the
    /// clamped components at zero, freezing them at their bounds.
    /// @param[in] current_gradient The gradient H x + g at the current iterate.
    /// @param[in] free_indices The free components at that same iterate.
    /// @returns The search direction, zero on the clamped components.
    [[nodiscard]] Vector newton_direction(const Gradient& current_gradient,
                                          const FreeIndices& free_indices) const;

    /// @brief The outcome of an accepted line search: the next iterate and its cost.
    struct LineSearchResult
    {
        /// The point the search settled on, which becomes the next iterate
        Vector next_point;
        /// Cost at next_point
        Scalar next_cost;
    };

    /// @brief Backtracking line search along the Newton direction.
    /// @details Every trial point is clamped back into the box, so a single step can activate
    /// several bounds at once. A step is accepted once its actual decrease is at least
    /// min_accept_ratio of the decrease the linear model predicts.
    /// @param[in] problem The problem data.
    /// @param[in] current_point The iterate the search starts from.
    /// @param[in] current_cost The cost at current_point.
    /// @param[in] direction The search direction.
    /// @param[in] expected_decrease The predicted decrease direction^T grad, which must be
    /// negative.
    /// @returns The next iterate and its value, or std::nullopt if no trial step was accepted.
    [[nodiscard]] std::optional<LineSearchResult> line_search(const Problem& problem,
                                                              const Vector& current_point,
                                                              Scalar current_cost,
                                                              const Vector& direction,
                                                              Scalar expected_decrease) const;

    /// Solver config
    BoxQPConfig<Scalar_T> config_;

    /// Cholesky factorization of the free-free sub-block of the Hessian
    Eigen::LLT<ReducedHessian> reduced_hessian_llt_;

    /// Number of iterations performed
    int iterations_{0};
};

}  // namespace ilqr::math

#include "box_active_set.inl"
