#pragma once

#include "trajectory.hpp"

namespace ilqr
{

template <typename Dims_T>
Trajectory<Dims_T>::Trajectory(int horizon_length)
{
    resize(horizon_length);
}

template <typename Dims_T>
void Trajectory<Dims_T>::resize(int horizon_length)
{
    states_.resize(horizon_length + 1);
    controls_.resize(horizon_length);
}

template <typename Dims_T>
int Trajectory<Dims_T>::horizon() const
{
    return static_cast<int>(controls_.size());
}

template <typename Dims_T>
auto Trajectory<Dims_T>::state(int i) -> StateVec&
{
    return states_.at(i);
}

template <typename Dims_T>
auto Trajectory<Dims_T>::state(int i) const -> const StateVec&
{
    return states_.at(i);
}

template <typename Dims_T>
auto Trajectory<Dims_T>::control(int i) -> ControlVec&
{
    return controls_.at(i);
}

template <typename Dims_T>
auto Trajectory<Dims_T>::control(int i) const -> const ControlVec&
{
    return controls_.at(i);
}

template <typename Dims_T>
auto Trajectory<Dims_T>::states() -> std::span<StateVec>
{
    return states_;
}

template <typename Dims_T>
auto Trajectory<Dims_T>::states() const -> std::span<const StateVec>
{
    return states_;
}

template <typename Dims_T>
auto Trajectory<Dims_T>::controls() -> std::span<ControlVec>
{
    return controls_;
}

template <typename Dims_T>
auto Trajectory<Dims_T>::controls() const -> std::span<const ControlVec>
{
    return controls_;
}
}  // namespace ilqr