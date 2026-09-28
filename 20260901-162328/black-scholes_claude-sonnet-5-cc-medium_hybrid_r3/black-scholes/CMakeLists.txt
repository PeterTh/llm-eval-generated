cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single translation unit as CUDA so it can host both the
# MPI/OpenMP driver code and the __global__ pricing kernel.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-Wall,-Wextra,-march=native,-fopenmp>
)

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
