#include "ilqr/core/trajectory.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace ilqr
{

using TestDims = Dims<2, 1, double>;

TEST(Trajectory, trajectory)
{
    auto trajectory = Trajectory<TestDims>();
    EXPECT_TRUE(true);
}
}  // namespace ilqr