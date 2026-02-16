cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
add_executable(black_scholes black_scholes.cpp)
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(black_scholes PRIVATE ${MPI_CXX_LIBRARIES} m)
