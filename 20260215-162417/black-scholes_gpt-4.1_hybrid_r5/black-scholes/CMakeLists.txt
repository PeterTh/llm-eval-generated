cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
enable_language(CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, and CUDA libraries
target_link_libraries(black_scholes m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)

target_compile_definitions(black_scholes PRIVATE USE_MPI USE_OPENMP USE_CUDA)
