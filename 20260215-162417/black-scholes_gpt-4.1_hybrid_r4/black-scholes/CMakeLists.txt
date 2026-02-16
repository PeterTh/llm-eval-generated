cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
target_link_libraries(black_scholes PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
