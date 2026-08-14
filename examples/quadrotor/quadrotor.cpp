// Planar quadrotor: an underactuated free-flyer solved by iLQR. It flies through a sequence of
// waypoints, chaining each solve from where the previous one ended, warm-started with hover thrust.

#include <fstream>
#include <ilqr/dynamics/autodiff_policy.hpp>
#include <ilqr/io/csv.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "example_io.hpp"
#include "model_definition.hpp"

namespace quadrotor::config
{
// Horizon length, time step and max. nr. of solver iterations
constexpr int horizon = 150;
constexpr double dt = 0.02;
constexpr int max_iterations = 500;

// Parameters
const QuadrotorParameters params{};
constexpr double gravity = 9.81;

// Initial state
const StateVec initial_state = StateVec::Zero();

// Cost weight diagonals: light pull toward the target waypoint, small control effort, heavy
// terminal penalty
const ControlVec control_weights{0.001, 0.001};
const StateVec target_running_weights{0.1, 0.1, 0.0, 0.0, 0.0, 0.0};
const StateVec target_final_weights{100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0};

// Flight waypoints x- and y-coordinates
const std::vector<Eigen::Vector<Scalar, 2>> waypoints = {{Scalar(5.0), Scalar(3.0)},
                                                         {Scalar(0.0), Scalar(10.0)},
                                                         {Scalar(-5.0), Scalar(3.0)},
                                                         {Scalar(0.0), Scalar(0.0)}};
}  // namespace quadrotor::config

int main()
{
    using namespace quadrotor;
    namespace cfg = quadrotor::config;

    // Discretize model with Heun and compute jacobians with automatic differentiation
    auto dyn = ilqr::discretize<Dims>(Quadrotor{cfg::params}, cfg::dt, ilqr::math::HeunStep{},
                                      ilqr::AutoDiff{});

    // Task-invariant control-effort cost
    ilqr::ControlPenaltyCost<Dims> control_cost(cfg::control_weights.asDiagonal());

    // Define solver configuration
    ilqr::SolverConfig<Scalar> config;
    config.max_iterations = cfg::max_iterations;

    // Warm start control: hover thrust that cancels gravity, split evenly across the two rotors.
    const double mass = cfg::params.m_body + 2 * cfg::params.m_rotor;
    const ControlVec hover_thrust{mass * cfg::gravity / 2.0, mass * cfg::gravity / 2.0};

    // The quadrotor chains from one waypoint to the next
    StateVec x_start = cfg::initial_state;

    // Write target waypoints
    std::ofstream targets_csv(ilqr::examples::output_path("quadrotor_targets.csv"));
    targets_csv << "idx,target_x,target_y\n";

    std::cout << "iLQR planar quadrotor — reaching a sequence of waypoints\n";
    for (std::size_t i = 0; i < cfg::waypoints.size(); ++i)
    {
        const auto& waypoint = cfg::waypoints[i];

        // Cost function to reach and hold the target waypoint: running pull on position, heavy
        // terminal on the full state (zero velocities and tilt at waypoint)
        const StateVec x_target{waypoint(0), waypoint(1), 0.0, 0.0, 0.0, 0.0};
        ilqr::QuadraticStateRegulatorCost<Dims> running_target_cost(
            cfg::target_running_weights.asDiagonal(), x_target);
        ilqr::FinalCost<Dims> final_target_cost(cfg::target_final_weights.asDiagonal(), x_target);
        ilqr::CompositeCostFunction cost(control_cost, running_target_cost, final_target_cost);

        // Instantiate solver
        ilqr::ILQRSolver solver{dyn, cost, config};

        // Create a solve request with warm start (initial control guess is hover thrust for gravity
        // compensation)
        const auto request = ilqr::SolveRequest<Dims>::warm_start(
            x_start, ilqr::AlignedVec<ControlVec>(cfg::horizon, hover_thrust));
        ilqr::SolveDiagnostics<double> diagnostics;

        // Solve with diagnostics
        const auto result = solver.solve(request, diagnostics);

        // Print and write results
        const StateVec x_final = result.trajectory.state(result.trajectory.horizon());
        const Eigen::Vector2d reached{x_final(0), x_final(1)};
        ilqr::examples::print_target_result(i, waypoint, reached, result, diagnostics);
        ilqr::write_trajectory_csv(
            ilqr::examples::output_path("quadrotor_trajectory_" + std::to_string(i) + ".csv"),
            result.trajectory, cfg::dt);
        targets_csv << i << ',' << waypoint(0) << ',' << waypoint(1) << '\n';

        // Start position for next task is the final position of this task
        x_start = x_final;
    }

    std::cout << "  visualize  : python examples/quadrotor/viz/animate_quadrotor.py\n";
    return 0;
}
