cmake_minimum_required(VERSION 3.23)

project(ilqr 
        VERSION 0.1.0
        DESCRIPTION "Iterative Linear-Quadratic Regulator (iLQR) optimization solver library"
        LANGUAGES CXX)

# Dependencies
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found — fetching Eigen 3.4.0")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR do-not-configure-eigen)
    FetchContent_MakeAvailable(Eigen3)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}")
endif()

# Header-only library — deliberately autodiff-agnostic. The optional ilqr::AutoDiff policy lives in
# a single opt-in header (ilqr/dynamics/autodiff_policy.hpp) that pulls in the third-party autodiff
# library itself; the core neither finds, fetches, nor links it, so the default build is
# dependency-light and needs no flag.
add_library(ilqr INTERFACE)
add_library(ilqr::ilqr ALIAS ilqr)

target_compile_features(ilqr INTERFACE cxx_std_20)
target_link_libraries(ilqr INTERFACE Eigen3::Eigen)

target_sources(ilqr
    INTERFACE
        FILE_SET ilqr_headers TYPE HEADERS
        BASE_DIRS
            include
        FILES
            include/ilqr/ilqr.hpp
            include/ilqr/core/config.hpp
            include/ilqr/core/solver.hpp
            include/ilqr/core/solver.inl
            include/ilqr/core/solver_diagnostics.hpp
            include/ilqr/core/solver_io.hpp
            include/ilqr/core/solver_io.inl
            include/ilqr/core/trajectory.hpp
            include/ilqr/core/trajectory.inl
            include/ilqr/core/types.hpp
            include/ilqr/cost/cost_concepts.hpp
            include/ilqr/cost/cost_concepts.inl
            include/ilqr/cost/composite_cost.hpp
            include/ilqr/cost/composite_cost.inl
            include/ilqr/cost/cost_terms.hpp
            include/ilqr/cost/cost_terms.inl
            include/ilqr/dynamics/dynamics_concepts.hpp
            include/ilqr/dynamics/finite_difference_policy.hpp
            include/ilqr/dynamics/autodiff_policy.hpp
            include/ilqr/dynamics/discretized_model.hpp
            include/ilqr/math/concepts.hpp
            include/ilqr/math/finite_differences.hpp
            include/ilqr/math/integrators.hpp
            include/ilqr/io/csv.hpp
)

# Options for tests and standalone examples - build them only when this is the top-level project
option(ILQR_BUILD_TESTS "Build the iLQR unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ILQR_BUILD_EXAMPLES "Build the iLQR examples" ${PROJECT_IS_TOP_LEVEL})

# Optional automatic differentiation for the demos, auto-detected (no flag, no fetch): when autodiff
# is installed/findable, the tests and examples that use ilqr::AutoDiff are built against it;
# otherwise they are skipped.
if(ILQR_BUILD_TESTS OR ILQR_BUILD_EXAMPLES)
    find_package(autodiff QUIET)
    if(autodiff_FOUND)
        message(STATUS "autodiff found — building the AutoDiff demos")
    else()
        message(STATUS "autodiff not found — skipping the AutoDiff demos "
                       "(install autodiff to enable them)")
    endif()
endif()

# Tests
if(ILQR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(ILQR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install and export - enable find_package(ilqr)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ilqr EXPORT ilqrTargets FILE_SET ilqr_headers)
install(EXPORT ilqrTargets
    FILE ilqrTargets.cmake
    NAMESPACE ilqr::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ilqr")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ilqrConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ilqrConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ilqr")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ilqrConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ilqrConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ilqrConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ilqr")