#pragma once

// Umbrella header: pulls in the full public iLQR API so consumers can simply
// #include <ilqr/ilqr.hpp> instead of the individual module headers.

#include "ilqr/core/config.hpp"
#include "ilqr/core/solver.hpp"
#include "ilqr/core/solver_diagnostics.hpp"
#include "ilqr/core/solver_io.hpp"
#include "ilqr/core/trajectory.hpp"
#include "ilqr/core/types.hpp"
#include "ilqr/cost/composite_cost.hpp"
#include "ilqr/cost/cost_concepts.hpp"
#include "ilqr/cost/cost_terms.hpp"
#include "ilqr/dynamics/discretized_model.hpp"
#include "ilqr/dynamics/dynamics_concepts.hpp"
#include "ilqr/dynamics/finite_difference_policy.hpp"

// NOTE: ilqr/dynamics/autodiff_policy.hpp is deliberately NOT included here — it is opt-in (it will
// pull in a third-party autodiff library). Include it explicitly to use ilqr::AutoDiff.

#include "ilqr/math/concepts.hpp"
#include "ilqr/math/finite_differences.hpp"
#include "ilqr/math/integrators.hpp"
