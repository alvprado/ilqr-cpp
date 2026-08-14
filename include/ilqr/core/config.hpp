#pragma once

#include <cmath>
#include <concepts>
#include <limits>

namespace ilqr
{

/// @brief Config parameters of the iLQR solver
template <std::floating_point Scalar = double>
struct SolverConfig
{
    /// Relative precision of the Scalar type
    static constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

    /// Maximum number of solver iterations before termination without convergence
    int max_iterations{100};

    /// @brief Convergence specifications - default values are precision-relative
    struct Convergence
    {
        /// Relative cost change tolerance to accept convergence
        Scalar cost_tol{Scalar(100) * eps};
        /// Feedforward/Gradient norm tolerance to accept convergence
        Scalar grad_tol{Scalar(1e-4)};
    } convergence;

    /// @brief Levenberg–Marquardt regularization config for ill-conditioned problems where the
    /// Hessians to be inversed are not positive-definite
    /// @details Used for being able to inverse Quu via Cholensky-Decomposition
    struct Regularization
    {
        /// The initial regularization value - no regularization unless problem requires it
        Scalar init{Scalar(0)};

        /// The minimal regularization value - the first regularization value to set if problem is
        /// ill-conditioned
        Scalar min{Scalar(1e-6)};

        /// The maximal regularization value
        Scalar max{Scalar(1e8)};

        /// The scaling factor - reg *= factor on failed iteration, reg /= on successfull iteration
        Scalar factor{Scalar(2)};

        /// The maximum number of trying regularization
        int max_iterations{50};
    } regularization;

    /// @brief Line search config parameters which regulates the control update given with
    /// u_new[k] = u[k] + α·k[k] + K[k]·(x_new[k] − x[k])
    /// α is the line search parameter. Starts at 1 and is reduced by αᵢ = ratio^i with i being the
    /// step number starting at 0
    struct LineSearch
    {
        /// Number of rollouts to try
        int steps{10};

        /// Ratio at which the line search parameter decays
        Scalar ratio{Scalar(0.5)};

        /// Minimum fraction of predicted reduction to accept a forward step
        Scalar min_accept_ratio{Scalar(1e-4)};
    } line_search;

    /// Verbose flag for solver diagnostics
    bool verbose{false};
};

}  // namespace ilqr
