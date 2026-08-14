# iLQR Solver

A header-only, templated **iterative Linear-Quadratic Regulator (iLQR)** solver for trajectory optimization in modern C++.

## Features

- **Header-only**, templated on problem dimensions and scalar type (`ilqr::Dims<nx, nu, Scalar>`).
- **Just write `ẋ = f(x, u)`** — choose a discretization policy (`EulerStep` / `HeunStep` / `RK4Step`)
  and a Jacobian policy (**finite differences** or **exact automatic differentiation**)
- **Composable costs** — You can provide your own cost function with its corresponding second-order Taylor expansion evaluation. A family of quadratic state regulator/tracker, control penalty and terminal cost are already provided.
- **Robust iLQR** — Tassa-style regularization, line-search in forward pass, cold/warm starts, per-iteration diagnostics, and CSV
  logging.
- Modern **C++20**

## Dependencies

| Dependency | Required | Notes |
|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) ≥ 3.4 | yes | auto-fetched if not installed |
| [autodiff](https://autodiff.github.io) | for the examples | header-only; enables the exact automatic-differentiation Jacobian policy. Auto-detected (not fetched) — install it to build the examples |
| [GoogleTest](https://github.com/google/googletest) | tests only | auto-fetched if missing |
| Python 3 + `matplotlib`, `numpy`, `pillow` | visualization | see `examples/viz/requirements.txt` |

A C++20 compiler is required.

## Build, test, run

```bash
cmake -S . -B build
cmake --build build -j

# unit tests
ctest --test-dir build

# run an example
./build/examples/cartpole/cartpole      # also: robotic_arm, quadrotor

# animate a run (writes a GIF into ./output/)
python examples/cartpole/viz/animate_cartpole.py --save cartpole.gif
```

To use the library from your own CMake project, `find_package(ilqr)` and link `ilqr::ilqr`, or pull
the repo in via `FetchContent` / `add_subdirectory`.

## At a glance

```cpp
#include <ilqr/ilqr.hpp>
#include <ilqr/dynamics/autodiff_policy.hpp>

using Dims = ilqr::Dims<4, 1>;  // 4 states, 1 control

// Continuous dynamics ẋ = f(x, u); templated scalar so autodiff can flow through.
struct CartPole {
    template <typename T>
    Eigen::Vector<T, 4> operator()(const Eigen::Vector<T, 4>& x,
                                   const Eigen::Vector<T, 1>& u) const;
};

int main() {
    // Model + discretization (RK4) + exact autodiff Jacobians.
    auto dyn = ilqr::discretize<Dims>(CartPole{}, /*dt=*/0.02,
                                      ilqr::math::RK4Step{}, ilqr::AutoDiff{});

    // Compose a cost from reusable terms (weights/goal omitted here).
    ilqr::CompositeCostFunction cost{
        ilqr::QuadraticStateRegulatorCost<Dims>(Q),
        ilqr::ControlPenaltyCost<Dims>(R),
        ilqr::FinalCost<Dims>(Qf, goal)};

    ilqr::ILQRSolver solver{dyn, cost, ilqr::SolverConfig<double>{}};
    auto request = ilqr::SolveRequest<Dims>::cold_start(x0, /*horizon=*/150);
    ilqr::SolveDiagnostics<double> diagnostics;
    auto result = solver.solve(request, diagnostics);   // result.trajectory, result.status
}
```

## Examples

Three worked examples live under `examples/`, each with a small Python animation.

### Cart-pole swing-up
Underactuated (a single force on the cart) — iLQR pumps energy to swing the pole upright.

<!-- Upload the GIF to GitHub (drag it into a comment/release) and paste its URL as src -->
<p align="center"><img src="REPLACE_WITH_CARTPOLE_GIF_URL" alt="Cart-pole swing-up" width="460"></p>

### 2-link robotic arm
Fully-actuated arm reaching a sequence of end-effector targets, chaining each reach into the next
via a custom task-space cost.

<p align="center"><img src="REPLACE_WITH_ROBOTIC_ARM_GIF_URL" alt="2-link robotic arm reaching" width="460"></p>

### Planar quadrotor
Underactuated free-flyer — it must tilt to translate, flying through a sequence of waypoints
warm-started with hover thrust.

<p align="center"><img src="REPLACE_WITH_QUADROTOR_GIF_URL" alt="Planar quadrotor flight" width="460"></p>

## License

MIT — see [LICENSE](LICENSE).
