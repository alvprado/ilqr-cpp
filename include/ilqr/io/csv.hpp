#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <string_view>

#include "ilqr/core/solver_diagnostics.hpp"
#include "ilqr/core/trajectory.hpp"

namespace ilqr
{

/// @brief Write a trajectory to CSV
/// @details Columns are `k,t,x0,..,x{nx-1},u0,..,u{nu-1}`. The state sequence has length N+1 and
/// the control sequence length N, so the terminal row's control cells are written as `nan`.
/// @tparam Dims_T Dimensions of the dynamic system (deduced from the trajectory).
/// @param path Output file path
/// @param trajectory The trajectory to write (states x_0..x_N, controls u_0..u_{N-1}).
/// @param dt Timestep, used to fill the time column t = k * dt.
template <typename Dims_T>
void write_trajectory_csv(std::string_view path, const Trajectory<Dims_T>& trajectory,
                          typename Dims_T::Scalar dt)
{
    constexpr int state_dim = Dims_T::state_dim;
    constexpr int control_dim = Dims_T::control_dim;
    const int N = trajectory.horizon();

    std::ofstream out{std::string{path}};
    out << std::setprecision(10);

    out << "k,t";
    for (int i = 0; i < state_dim; ++i) out << ",x" << i;
    for (int j = 0; j < control_dim; ++j) out << ",u" << j;
    out << '\n';

    for (int k = 0; k <= N; ++k)
    {
        out << k << ',' << k * dt;
        const auto& state = trajectory.state(k);
        for (int i = 0; i < state_dim; ++i) out << ',' << state(i);

        // Controls have one less entry than states - the terminal row has no control
        if (k < N)
        {
            const auto& control = trajectory.control(k);
            for (int j = 0; j < control_dim; ++j) out << ',' << control(j);
        }
        else
        {
            for (int j = 0; j < control_dim; ++j) out << ",nan";
        }
        out << '\n';
    }
}

/// @brief Write per-iteration solver diagnostics to CSV.
/// @details Columns are `iter,cost,reg,grad_norm,step_size,accepted` — one row per outer iteration
/// @tparam Scalar_T Diagnostics scalar type (deduced).
/// @param path Output file path.
/// @param diagnostics The per-iteration diagnostics to write.
template <std::floating_point Scalar_T>
void write_diagnostics_csv(std::string_view path, const SolveDiagnostics<Scalar_T>& diagnostics)
{
    std::ofstream out{std::string{path}};
    out << std::setprecision(10) << "iter,cost,reg,grad_norm,step_size,accepted\n";
    for (std::size_t i = 0; i < diagnostics.iterations.size(); ++i)
    {
        const auto& record = diagnostics.iterations[i];
        out << i << ',' << record.cost << ',' << record.reg << ',' << record.grad_norm << ','
            << record.step_size << ',' << (record.accepted_forward_pass ? 1 : 0) << '\n';
    }
}

}  // namespace ilqr
