#include <algorithm>  // std::max
#include <cmath>      // std::abs, std::isfinite

#include "box_active_set.hpp"

namespace ilqr::math
{

template <int N, std::floating_point Scalar_T>
BoxQPActiveSetMethod<N, Scalar_T>::BoxQPActiveSetMethod(const BoxQPConfig<Scalar_T>& config)
    : config_(config)
{
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::evaluate_cost(const Problem& problem, const Vector& x)
    -> Scalar
{
    return x.dot(problem.gradient) + Scalar(0.5) * x.dot(problem.hessian * x);
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::clamp_into_box(const Problem& problem, const Vector& x)
    -> Vector
{
    return x.cwiseMax(problem.lower_bound).cwiseMin(problem.upper_bound);
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::compute_free_indices(const Problem& problem,
                                                             const Vector& x,
                                                             const Gradient& current_gradient)
    -> FreeIndices
{
    FreeIndices free_indices(N);
    int free_count = 0;

    for (int component = 0; component < N; ++component)
    {
        const bool at_lower_bound = x(component) <= problem.lower_bound(component);
        const bool at_upper_bound = x(component) >= problem.upper_bound(component);

        const bool is_clamped = (at_lower_bound && current_gradient(component) > Scalar(0)) ||
                                (at_upper_bound && current_gradient(component) < Scalar(0));

        if (!is_clamped)
        {
            free_indices(free_count++) = component;
        }
    }

    free_indices.conservativeResize(free_count);
    return free_indices;
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::newton_direction(const Gradient& current_gradient,
                                                         const FreeIndices& free_indices) const
    -> Vector
{
    const ReducedGradient reduced_step = reduced_hessian_llt_.solve(current_gradient(free_indices));

    // Clamped components stay at zero, freezing them on their bounds.
    Vector direction = Vector::Zero();
    direction(free_indices) = -reduced_step;
    return direction;
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::line_search(const Problem& problem,
                                                    const Vector& current_point,
                                                    Scalar current_cost, const Vector& direction,
                                                    Scalar expected_decrease) const
    -> std::optional<LineSearchResult>
{
    Scalar step_size = Scalar(1);

    for (int trial_step = 0; trial_step < config_.line_search.steps; ++trial_step)
    {
        const Vector candidate_point =
            clamp_into_box(problem, current_point + step_size * direction);
        const Scalar candidate_cost = evaluate_cost(problem, candidate_point);

        if (candidate_cost - current_cost <=
            config_.line_search.min_accept_ratio * step_size * expected_decrease)
        {
            return LineSearchResult{.next_point = candidate_point, .next_cost = candidate_cost};
        }

        // Compute line search step size alpha_i = ratio^i
        step_size *= config_.line_search.ratio;
    }

    return std::nullopt;
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::solve(const Hessian& hessian, const Gradient& gradient,
                                              const Vector& lower_bound, const Vector& upper_bound,
                                              const Vector& initial_guess) -> std::optional<Result>
{
    // Bundle problem
    const Problem problem{hessian, gradient, lower_bound, upper_bound};

    // Initialize current point and value
    Vector current_point = clamp_into_box(problem, initial_guess);
    Scalar current_cost = evaluate_cost(problem, current_point);
    iterations_ = 0;

    // Free indices
    FreeIndices free_indices;

    // Previous free indices used to decide whether a new factorization is needed or not depending
    // if the active set changed after one iteration
    FreeIndices previous_free_indices;

    // Helper to gather a result
    const auto make_result = [&]() -> Result
    {
        return Result{.optimal_vector = current_point,
                      .free_indices = free_indices,
                      .factorized_free_hessian = reduced_hessian_llt_};
    };

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration)
    {
        const Gradient current_gradient = gradient + hessian * current_point;
        free_indices = compute_free_indices(problem, current_point, current_gradient);

        // If there is no free component, this is the constrained optimum and there is no free
        // subspace left to factorize
        if (free_indices.size() == 0)
        {
            reduced_hessian_llt_ = Eigen::LLT<ReducedHessian>{};
            return make_result();
        }

        // Only factorize if the free subspace changed
        const bool free_set_moved = free_indices.size() != previous_free_indices.size() ||
                                    (free_indices != previous_free_indices).any();
        if (free_set_moved)
        {
            // Cholesky-factorize the free-free subblock of the hessian
            reduced_hessian_llt_.compute(hessian(free_indices, free_indices));
            if (reduced_hessian_llt_.info() != Eigen::Success)
            {
                return std::nullopt;
            }
            previous_free_indices = free_indices;
        }

        // Return if the projected gradient on the free indices (KKT residual) is smaller than the
        // given tolerance
        if (current_gradient(free_indices).cwiseAbs().maxCoeff() < config_.convergence.grad_tol)
        {
            return make_result();
        }

        // Compute newton direction and expected decrease
        const Vector direction = newton_direction(current_gradient, free_indices);
        const Scalar expected_decrease = direction.dot(current_gradient);

        // Cannot happen with a positive-definite free Hessian, but a badly scaled problem can
        // round its way here.
        if (expected_decrease >= Scalar(0))
        {
            return std::nullopt;
        }

        // Compute line search
        const auto line_search_result =
            line_search(problem, current_point, current_cost, direction, expected_decrease);
        if (!line_search_result.has_value())
        {
            return std::nullopt;
        }

        const Scalar relative_improvement = (current_cost - line_search_result->next_cost) /
                                            std::max(std::abs(current_cost), config_.eps);

        current_point = line_search_result->next_point;
        current_cost = line_search_result->next_cost;
        ++iterations_;

        // Return if the cost has stopped moving
        if (relative_improvement < config_.convergence.rel_cost_improvement_tol)
        {
            return make_result();
        }
    }

    // Iteration limit has been reached without converging
    return std::nullopt;
}

template <int N, std::floating_point Scalar_T>
auto BoxQPActiveSetMethod<N, Scalar_T>::solve(const Hessian& hessian, const Gradient& gradient,
                                              const Vector& lower_bound, const Vector& upper_bound)
    -> std::optional<Result>
{
    // Midpoint of the bounds, with infinite (or doubly-infinite) intervals falling back to zero.
    Vector initial_guess;
    for (int component = 0; component < N; ++component)
    {
        const Scalar midpoint = (lower_bound(component) + upper_bound(component)) / Scalar(2);
        initial_guess(component) = std::isfinite(midpoint) ? midpoint : Scalar(0);
    }

    return solve(hessian, gradient, lower_bound, upper_bound, initial_guess);
}

}  // namespace ilqr::math
