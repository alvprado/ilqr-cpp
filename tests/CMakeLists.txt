find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

foreach(test_name IN ITEMS solver_test cost_test trajectory_test discretization_test)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE ilqr::ilqr GTest::gtest_main)
    gtest_discover_tests(${test_name})
endforeach()

if(autodiff_FOUND)
    add_executable(autodiff_test autodiff_test.cpp)
    target_link_libraries(autodiff_test PRIVATE ilqr::ilqr GTest::gtest_main autodiff::autodiff)
    gtest_discover_tests(autodiff_test)
endif()
