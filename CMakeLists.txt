cmake_minimum_required(VERSION 3.24)
project(matmul_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Benchmarks only make sense in an optimized build, so default to Release.
# For GCC, CMake's Release configuration already adds "-O3 -DNDEBUG".
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MATMUL_NATIVE "Compile with -march=native (turn OFF for portable/CI builds)" ON)
option(MATMUL_BUILD_BENCH "Build Google Benchmark executables" ON)

message(STATUS "matmul-bench: build type = ${CMAKE_BUILD_TYPE}, MATMUL_NATIVE = ${MATMUL_NATIVE}")

# Flags shared by all of OUR targets. Fetched dependencies keep their own flags.
add_library(matmul_flags INTERFACE)
target_compile_options(matmul_flags INTERFACE -Wall -Wextra -Wpedantic)
if(MATMUL_NATIVE)
  target_compile_options(matmul_flags INTERFACE -march=native)
endif()

# ---- Dependencies -----------------------------------------------------------
include(FetchContent)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(MATMUL_BUILD_BENCH)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

# ---- Tests ------------------------------------------------------------------
enable_testing()
include(GoogleTest)

add_executable(matmul_tests
  tests/test_hello.cpp)
target_link_libraries(matmul_tests PRIVATE matmul_flags GTest::gtest_main)
gtest_discover_tests(matmul_tests)

# ---- Benchmarks -------------------------------------------------------------
if(MATMUL_BUILD_BENCH)
  add_executable(matmul_bench
    bench/bench_bandwidth.cpp)
  target_link_libraries(matmul_bench PRIVATE matmul_flags benchmark::benchmark_main)
endif()
