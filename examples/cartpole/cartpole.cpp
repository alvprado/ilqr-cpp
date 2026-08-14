// Cart-pole swing-up: a nonlinear, underactuated system driven upright by iLQR.

#include <ilqr/dynamics/autodiff_policy.hpp>
#include <ilqr/io/csv.hpp>
#include <iostream>
#include <numbers>

#include "example_io.hpp"
#include "model_definition.hpp"

namespace cartpole::config
{
// Define a 3 seconds trajectory with 0.02 seconds time step
constexpr int horizon = 150;
constexpr double dt = 0.02;

// Max. number of solver iterations
constexpr int max_iterations = 500;

// Cart centered with the pole hanging down; goal is upright, centered, at rest.
const StateVec initial_state{0.0, std::numbers::pi, 0.0, 0.0};
const StateVec goal_state = StateVec::Zero();

// Cost weight diagonals: light running pull toward upright, small control effort, heavy terminal
// penalty
const StateVec running_state_weights{0.1, 0.1, 0.001, 0.001};
const ControlVec control_weights{0.1};
const StateVec final_state_weights{100.0, 100.0, 10.0, 10.0};
}  // namespace cartpole::config

int main()
{
    using namespace cartpole;
    namespace cfg = cartpole::config;

    // Discretize model with Euler and compute jacobians with automatic differentiation
    auto dyn =
        ilqr::discretize<Dims>(CartPole{}, cfg::dt, ilqr::math::EulerStep{}, ilqr::AutoDiff{});

    // Define and compose cost terms
    ilqr::QuadraticStateRegulatorCost<Dims> running_cost(cfg::running_state_weights.asDiagonal());
    ilqr::ControlPenaltyCost<Dims> control_cost(cfg::control_weights.asDiagonal());
    ilqr::FinalCost<Dims> terminal_cost(cfg::final_state_weights.asDiagonal(), cfg::goal_state);
    ilqr::CompositeCostFunction cost(running_cost, control_cost, terminal_cost);

    // Define solver configuration
    ilqr::SolverConfig<double> config;
    config.max_iterations = cfg::max_iterations;

    // Instantiate solver
    ilqr::ILQRSolver solver{dyn, cost, config};

    // Create a solve request with cold start (initial control guess is zero for the entire horizon)
    const auto request = ilqr::SolveRequest<Dims>::cold_start(cfg::initial_state, cfg::horizon);
    ilqr::SolveDiagnostics<double> diagnostics;

    // Solve
    const auto result = solver.solve(request, diagnostics);

    // Print and write results
    ilqr::examples::print_summary("iLQR cart-pole swing-up", result, diagnostics, cfg::goal_state);
    ilqr::write_trajectory_csv(ilqr::examples::output_path("cartpole_trajectory.csv"),
                               result.trajectory, cfg::dt);
    ilqr::write_diagnostics_csv(ilqr::examples::output_path("cartpole_diagnostics.csv"),
                                diagnostics);
    std::cout << "  visualize  : python examples/cartpole/viz/animate_cartpole.py\n";
    return 0;
}
