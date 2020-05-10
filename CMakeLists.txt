cmake_minimum_required(VERSION 3.13)
project(Stats)

set(CMAKE_CXX_STANDARD 17)
set(EIGEN_DIR /usr/local/include/eigen3)
include_directories(${EIGEN_DIR})
include_directories(include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Must include to avoid failing on Linux with std::thread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(gtest REQUIRED)
find_package(benchmark REQUIRED)
find_package(GSL REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

# Options
option(BUILD_TESTS "Build Tests" ON)
option(BENCH_TESTS "Benchmark Tests" ON)

add_subdirectory(benchmarks)
add_subdirectory(test)
