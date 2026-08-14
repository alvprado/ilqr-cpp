add_library(example_common INTERFACE)
target_include_directories(example_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(cartpole)
add_subdirectory(robotic_arm)
add_subdirectory(quadrotor)
