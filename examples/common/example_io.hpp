#pragma once

// Small output helpers shared by the examples: a console summary of a solve, a one-line per-target
// result for the multi-reach examples, and an output-path helper that confines generated CSVs to an
// `output/` directory.

#include <concepts>
#include <cstddef>
#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "ilqr/core/solver_diagnostics.hpp"
#include "ilqr/core/solver_io.hpp"

namespace ilqr::examples
{

/// @brief Compact one-line format for printing a state vector as "[a, b, c]".
inline const Eigen::IOFormat& vector_format()
{
    static const Eigen::IOFormat format(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "",
                                        "", "[", "]");
    return format;
}

/// @brief Return "output/<filename>", creating the output directory if it does not exist. Keeps
/// generated CSVs out of the run directory root.
inline std::string output_path(std::string_view filename)
{
    std::filesystem::create_directories("output");
    return "output/" + std::string(filename);
}

/// @brief Final trajectory cost recorded in the diagnostics (0 if no iterations ran).
template <std::floating_point Scalar_T>
Scalar_T final_cost(const SolveDiagnostics<Scalar_T>& diagnostics)
{
    return diagnostics.iterations.empty() ? Scalar_T(0) : diagnostics.iterations.back().cost;
}

/// @brief Print a single-solve summary: status, iterations, final cost, and goal vs reached state.
template <typename Dims_T, std::floating_point Scalar_T>
void print_summary(std::string_view title, const Result<Dims_T>& result,
                   const SolveDiagnostics<Scalar_T>& diagnostics,
                   const typename Dims_T::StateVec& goal)
{
    const auto& trajectory = result.trajectory;
    const auto reached = trajectory.state(trajectory.horizon());
    std::cout << title << "\n"
              << "  status     : " << to_string(result.status) << "\n"
              << "  iterations : " << diagnostics.iterations.size() << "\n"
              << "  final cost : " << final_cost(diagnostics) << "\n"
              << "  goal state : " << goal.transpose().format(vector_format()) << "\n"
              << "  reached    : " << reached.transpose().format(vector_format()) << "\n";
}

/// @brief Print a one-line result for a single target in a multi-reach loop (arm, quadrotor).
template <typename Dims_T, std::floating_point Scalar_T>
void print_target_result(std::size_t index, const Eigen::Vector2d& target,
                         const Eigen::Vector2d& reached, const Result<Dims_T>& result,
                         const SolveDiagnostics<Scalar_T>& diagnostics)
{
    std::cout << "  goal state " << index << " = " << target.transpose().format(vector_format())
              << " | reached state = " << reached.transpose().format(vector_format())
              << " | status = " << to_string(result.status)
              << " | iters = " << diagnostics.iterations.size() << "\n";
}

}  // namespace ilqr::examples
