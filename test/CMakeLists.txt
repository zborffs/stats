# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic") # -Werror
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3")
endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(StatsTest stats_test.cpp)
add_executable(BootstrapTest bootstrap_test.cpp)
add_executable(ChessTest chess_test.cpp)

target_link_libraries(StatsTest Threads::Threads ${GTEST_LIBRARIES} GSL::gsl GSL::gslcblas)
target_link_libraries(BootstrapTest Threads::Threads ${GTEST_LIBRARIES})
target_link_libraries(ChessTest Threads::Threads ${GTEST_LIBRARIES})

add_test(StatsTest StatsTest)
add_test(BootstrapTest BootstrapTest)
add_test(ChessTest ChessTest)