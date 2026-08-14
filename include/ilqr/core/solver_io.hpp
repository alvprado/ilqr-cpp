#pragma once

#include <cstdint>
#include <vector>

#include "ilqr/core/trajectory.hpp"
#include "ilqr/core/types.hpp"

namespace ilqr
{

/// @brief An iLQR solve request
template <typename Dims_T>
class SolveRequest
{
public:
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;

    /// @brief Build a warm-started problem; the horizon is the length of the guess.
    /// @param initial_state The initial state x0.
    /// @param initial_controls_guess The initial control guess; must be non-empty, its length is N.
    [[nodiscard]] static SolveRequest warm_start(StateVec initial_state,
                                                 AlignedVec<ControlVec> initial_controls_guess);

    /// @brief Build a cold-started problem; the N controls are seeded to zero.
    /// @param initial_state The initial state x0.
    /// @param horizon The number of steps N (must be > 0).
    [[nodiscard]] static SolveRequest cold_start(StateVec initial_state, int horizon);

    /// @brief The horizon length N (equivalently, the number of controls).
    [[nodiscard]] int horizon() const;

    /// @brief The initial state x0.
    [[nodiscard]] const StateVec& initial_state() const;

    /// @brief The initial control guess, always of length N.
    [[nodiscard]] const AlignedVec<ControlVec>& initial_controls() const;

private:
    /// @brief Private Ctor
    /// @param initial_state The initial state x0
    /// @param controls The initial controls
    SolveRequest(StateVec initial_state, AlignedVec<ControlVec> controls);

    /// Initial state x0
    StateVec initial_state_;

    /// Initial controls - either zero for cold or guessed for warm starts
    AlignedVec<ControlVec> controls_;
};

/// @brief The iLQR solver status
enum class SolverStatus : std::uint8_t
{
    Converged = 0U,
    MaxIterations = 1U,
    MaxRegularization = 2U,
    InvalidProblem = 3U,
};

/// @brief SolverStatus to string
/// @param status The status to stringify
/// @return A static, null-terminated string; "Unknown" for an unrecognized value.
[[nodiscard]] inline const char* to_string(SolverStatus status)
{
    switch (status)
    {
        case SolverStatus::Converged:
            return "Converged";
        case SolverStatus::MaxIterations:
            return "MaxIterations";
        case SolverStatus::MaxRegularization:
            return "MaxRegularization";
        case SolverStatus::InvalidProblem:
            return "InvalidProblem";
    }
    return "Unknown";
}

/// @brief The iLQR solver result from which we obtain an optimized trajectory (x_k, u_k), the
/// optimal time-varying iLQR control policy K[k] and the solver status
template <typename Dims_T>
struct Result
{
    using ControlStateMat = typename Dims_T::ControlStateMat;

    /// Optimal trajectory
    Trajectory<Dims_T> trajectory;

    /// Optimal control policy
    AlignedVec<ControlStateMat> feedback_gains;

    /// Solver status
    SolverStatus status;
};

}  // namespace ilqr

#include "solver_io.inl"
