cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Compile source as CUDA to enable GPU kernels + host code in one translation unit
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimisations, vectorization, and OpenMP (passed through to host compiler)
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler -fopenmp)

# Link required libraries
find_package(MPI REQUIRED)
target_link_libraries(black_scholes PUBLIC MPI::MPI_CXX m -fopenmp)

# Target broad GPU compatibility (Volta/Turing/Ampere/Ada/Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90")
