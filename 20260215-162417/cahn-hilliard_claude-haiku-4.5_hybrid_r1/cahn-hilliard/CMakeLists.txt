cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA support is included in the code when available
# To enable CUDA compilation:
#   1. Ensure CUDA toolkit is installed
#   2. Set CMAKE_CUDA_COMPILER to nvcc (e.g., cmake -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc ..)
#   3. Remove DISABLE_CUDA from target_compile_definitions
# The code will unconditionally use CUDA acceleration when compiled with CUDA support

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Disable CUDA at compile time for default build (no CUDA toolkit required)
target_compile_definitions(cahn_hilliard PRIVATE DISABLE_CUDA)

# Link libraries: MPI, OpenMP, math library
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
