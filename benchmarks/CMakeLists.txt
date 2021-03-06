if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic") # -Werror
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3")
endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(MeanBench mean_bench.cpp)
add_executable(StdDevBench stddev_bench.cpp)
add_executable(VarBench var_bench.cpp)
add_executable(MaxBench max_bench.cpp)
add_executable(MinBench min_bench.cpp)
add_executable(MedianBench median_bench.cpp)

target_link_libraries(MeanBench benchmark::benchmark GSL::gsl GSL::gslcblas)
target_link_libraries(StdDevBench benchmark::benchmark GSL::gsl GSL::gslcblas)
target_link_libraries(VarBench benchmark::benchmark GSL::gsl GSL::gslcblas)
target_link_libraries(MaxBench benchmark::benchmark GSL::gsl GSL::gslcblas)
target_link_libraries(MinBench benchmark::benchmark GSL::gsl GSL::gslcblas)
target_link_libraries(MedianBench benchmark::benchmark GSL::gsl GSL::gslcblas)

add_test(MeanBench MeanBench)
add_test(StdDevBench StdDevBench)
add_test(VarBench VarBench)
add_test(MaxBench MaxBench)
add_test(MinBench MinBench)
add_test(MedianBench MedianBench)