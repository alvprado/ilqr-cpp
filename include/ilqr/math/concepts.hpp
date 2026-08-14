#pragma once

#include <concepts>
#include <Eigen/Dense>

namespace ilqr::math
{

/// @brief An Eigen column vector: a single-column MatrixBase expression. Exposes ::Scalar.
/// @tparam Vector_T The type to check.
template <typename Vector_T>
concept ColumnVector = std::derived_from<Vector_T, Eigen::MatrixBase<Vector_T>> &&
                       (Vector_T::ColsAtCompileTime == 1);

/// @brief A callable f(x, u) mapping a state and a control to a state-space vector.
/// @details The lower-level, dimension-agnostic contract the integrators and differentiation
/// policies build on. It fits both a continuous model (ẋ = f(x, u)) and a discrete step
/// (x_{k+1} = f(x, u)) — both share the (state, control) -> state signature.
/// @tparam Fn A callable invoked as f(x, u).
/// @tparam StateVec_T The state (column) vector type; f returns something convertible to it.
/// @tparam ControlVec_T The control (column) vector type.
template <typename Fn, typename StateVec_T, typename ControlVec_T>
concept StateTransition = requires(const Fn& f, const StateVec_T& x, const ControlVec_T& u) {
                              {
                                  f(x, u)
                              } -> std::convertible_to<StateVec_T>;
                          };

}  // namespace ilqr::math
