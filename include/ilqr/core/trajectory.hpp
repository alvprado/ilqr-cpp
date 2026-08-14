#pragma once

#include <span>

#include "ilqr/core/types.hpp"

namespace ilqr
{

/// @brief A discrete-time trajectory: the state and control sequences produced
///        by the dynamics x_{k+1} = f(x_k, u_k) over a horizon of N steps.
/// @tparam Dims_T Supplies the per-step vector types StateVec and ControlVec.
/// @note The state sequence has length N+1 (x_0 … x_N) and the control sequence
///       has length N (u_0 … u_{N-1}); each u_k drives x_k to x_{k+1}, so the
///       states are always one longer than the controls.
/// @note The two sequences are kept consistent: at any point they are either
///       both empty (horizon 0), or sized N+1 and N.
template <typename Dims_T>
class Trajectory
{
public:
    /// Aliases
    using Dims = Dims_T;
    using StateVec = typename Dims_T::StateVec;
    using ControlVec = typename Dims_T::ControlVec;

    /// @brief Default ctor - states and controls are empty
    Trajectory() = default;

    /// @brief Ctor taking an horizon length and resizing the trajectory
    /// @param horizon_length The number of steps in a horizon
    explicit Trajectory(int horizon_length);

    /// @brief Get horizon length
    [[nodiscard]] int horizon() const;

    /// @brief Resize the state and control containers
    /// @param horizon_length The number of steps in a horizon
    void resize(int horizon_length);

    /// @brief Access the i-th state x_i (mutable).
    /// @param i Step index in [0, N].
    /// @return Reference to x_i.
    [[nodiscard]] StateVec& state(int i);

    /// @brief Access the i-th state x_i (read-only).
    /// @param i Step index in [0, N].
    /// @return Const reference to x_i.
    [[nodiscard]] const StateVec& state(int i) const;

    /// @brief Access the i-th control u_i (mutable).
    /// @param i Step index in [0, N-1].
    /// @return Reference to u_i.
    [[nodiscard]] ControlVec& control(int i);

    /// @brief Access the i-th control u_i (read-only).
    /// @param i Step index in [0, N-1].
    /// @return Const reference to u_i.
    [[nodiscard]] const ControlVec& control(int i) const;

    /// @brief View over the full state sequence x_0 … x_N for iteration (mutable).
    /// @return Span of length N+1; empty if the trajectory is empty.
    [[nodiscard]] std::span<StateVec> states();

    /// @brief View over the full state sequence x_0 … x_N for iteration (read-only).
    /// @return Span of length N+1; empty if the trajectory is empty.
    [[nodiscard]] std::span<const StateVec> states() const;

    /// @brief View over the full control sequence u_0 … u_{N-1} for iteration (mutable).
    /// @return Span of length N; empty if the trajectory is empty.
    [[nodiscard]] std::span<ControlVec> controls();

    /// @brief View over the full control sequence u_0 … u_{N-1} for iteration (read-only).
    /// @return Span of length N; empty if the trajectory is empty.
    [[nodiscard]] std::span<const ControlVec> controls() const;

private:
    /// Container of states of size N+1 if not empty
    AlignedVec<StateVec> states_;

    /// Container of controls of size N if not empty
    AlignedVec<ControlVec> controls_;
};

}  // namespace ilqr

#include "trajectory.inl"
