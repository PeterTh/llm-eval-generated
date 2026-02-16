cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Compile with optimizations and enable OpenMP for host
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

# Link MPI, OpenMP and CUDA runtime
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDAToolkit::cudart m)
