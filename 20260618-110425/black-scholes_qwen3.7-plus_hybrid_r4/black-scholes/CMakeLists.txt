cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architecture (RTX 3090 = 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Treat the .cpp file as CUDA source so nvcc compiles it (supports __global__, __device__, etc.)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE 
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler;-O3,-march=native,-fopenmp;-lineinfo>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fopenmp;-Wall;-Wextra>"
)

# Link libraries
target_link_libraries(black_scholes 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    m
)
