#pragma once

// ilqr/solver_io.inl — template implementations for the SolveRequest factories and accessors declared
// in ilqr/solver_io.hpp. Not a standalone header: it is included at the bottom of solver_io.hpp.

#include <cassert>
#include <cstddef>
#include <utility>

#include "solver_io.hpp"

namespace ilqr
{

template <typename Dims_T>
SolveRequest<Dims_T>::SolveRequest(StateVec initial_state, AlignedVec<ControlVec> controls)
    : initial_state_(std::move(initial_state)), controls_(std::move(controls))
{
}

template <typename Dims_T>
SolveRequest<Dims_T> SolveRequest<Dims_T>::warm_start(StateVec initial_state,
                                            AlignedVec<ControlVec> initial_controls_guess)
{
    assert(!initial_controls_guess.empty() &&
           "warm_start needs a non-empty control guess; use cold_start for a zero guess");
    return SolveRequest(std::move(initial_state), std::move(initial_controls_guess));
}

template <typename Dims_T>
SolveRequest<Dims_T> SolveRequest<Dims_T>::cold_start(StateVec initial_state, int horizon)
{
    assert(horizon > 0 && "horizon must be positive");
    return SolveRequest(std::move(initial_state),
                   AlignedVec<ControlVec>(static_cast<std::size_t>(horizon), ControlVec::Zero()));
}

template <typename Dims_T>
int SolveRequest<Dims_T>::horizon() const
{
    return static_cast<int>(controls_.size());
}

template <typename Dims_T>
auto SolveRequest<Dims_T>::initial_state() const -> const StateVec&
{
    return initial_state_;
}

template <typename Dims_T>
auto SolveRequest<Dims_T>::initial_controls() const -> const AlignedVec<ControlVec>&
{
    return controls_;
}

}  // namespace ilqr
