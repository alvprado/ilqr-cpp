// 2-link planar robot arm: a fully-actuated reaching task solved by iLQR. The arm visits a sequence
// of end-effector targets, chaining each solve from where the previous one ended.

#include <cmath>
#include <fstream>
#include <ilqr/dynamics/autodiff_policy.hpp>
#include <ilqr/io/csv.hpp>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include "end_effector_cost.hpp"
#include "example_io.hpp"
#include "model_definition.hpp"

namespace robotic_arm::config
{
// Horizon length, time step and max. nr. of iterations
constexpr int horizon = 150;
constexpr double dt = 0.02;
constexpr int max_iterations = 500;

// Robot parameters
const RobotParameters params{};

// Initial state - the robot arms are horizontally aligend and folded
const StateVec initial_state{0.0, std::numbers::pi, 0.0, 0.0};

// Cost weight diagonals: small torques, small running joint velocity, heavy terminal velocity.
const ControlVec control_weights{0.1, 0.1};
const StateVec running_state_weights{0.0, 0.0, 0.01, 0.01};
const StateVec final_state_weights{0.0, 0.0, 100.0, 100.0};

// End-effector reaching weights
constexpr Scalar ee_running_weight = 1.0;
constexpr Scalar ee_final_weight = 1000.0;

// Reachable end-effector targets
const std::vector<Eigen::Vector<Scalar, 2>> targets = {
    {Scalar(1.0), Scalar(1.0)},
    {Scalar(-1.0), Scalar(1.0)},
    {Scalar(-1.0), Scalar(-1.0)},
    {Scalar(1.0), Scalar(-1.0)},
    {Scalar(params.l1 - params.l2), Scalar(0.0)}};
}  // namespace robotic_arm::config

int main()
{
    using namespace robotic_arm;
    namespace cfg = robotic_arm::config;

    // Discretize model with Runge-Kutta and compute jacobians with automatic differentiation
    auto dyn = ilqr::discretize<Dims>(RoboticArm{cfg::params}, cfg::dt, ilqr::math::RK4Step{},
                                      ilqr::AutoDiff{});

    // Task-invariant costs shared by every reach.
    ilqr::ControlPenaltyCost<Dims> control_cost(cfg::control_weights.asDiagonal());
    ilqr::QuadraticStateRegulatorCost<Dims> joint_velocity_cost(
        cfg::running_state_weights.asDiagonal());
    ilqr::FinalCost<Dims> joint_velocity_final_cost(cfg::final_state_weights.asDiagonal(),
                                                    StateVec::Zero());
    ilqr::CompositeCostFunction task_invariant_costs(control_cost, joint_velocity_cost,
                                                     joint_velocity_final_cost);

    // Define solver configuration
    ilqr::SolverConfig<Scalar> config;
    config.max_iterations = cfg::max_iterations;

    // The arm chains from one target to the next: each solve starts where the previous ended.
    StateVec x_start = cfg::initial_state;

    // Write target positions
    std::ofstream targets_csv(ilqr::examples::output_path("robotic_arm_targets.csv"));
    targets_csv << "idx,target_x,target_y\n";

    std::cout << "iLQR 2-link robotic arm — reaching a sequence of targets\n";
    for (std::size_t i = 0; i < cfg::targets.size(); ++i)
    {
        const auto& target = cfg::targets[i];

        // Cost function to reach the given target positions on the end-effector task space
        EndEffectorPositionCostConfig const ee_config{cfg::params, target(0), target(1),
                                                      cfg::ee_running_weight, cfg::ee_final_weight};
        EndEffectorPositionCost<Dims> end_effector_cost(ee_config);
        ilqr::CompositeCostFunction cost(task_invariant_costs, end_effector_cost);

        // Instantiate solver
        ilqr::ILQRSolver solver{dyn, cost, config};

        // Create a solve request with cold start (zero initial control guess)
        const auto request = ilqr::SolveRequest<Dims>::cold_start(x_start, cfg::horizon);
        ilqr::SolveDiagnostics<double> diagnostics;

        // Solve with diagnostics
        const auto result = solver.solve(request, diagnostics);

        // Post-processing steps
        const StateVec x_final = result.trajectory.state(result.trajectory.horizon());
        const Eigen::Vector2d reached{cfg::params.l1 * std::cos(x_final(0)) +
                                          cfg::params.l2 * std::cos(x_final(0) + x_final(1)),
                                      cfg::params.l1 * std::sin(x_final(0)) +
                                          cfg::params.l2 * std::sin(x_final(0) + x_final(1))};

        // Print and write results
        ilqr::examples::print_target_result(i, target, reached, result, diagnostics);
        ilqr::write_trajectory_csv(
            ilqr::examples::output_path("robotic_arm_trajectory_" + std::to_string(i) + ".csv"),
            result.trajectory, cfg::dt);
        targets_csv << i << ',' << target(0) << ',' << target(1) << '\n';

        // Start position for next reach is the final position of this task
        x_start = x_final;
    }

    std::cout << "  visualize  : python examples/robotic_arm/viz/animate_robotic_arm.py\n";
    return 0;
}
