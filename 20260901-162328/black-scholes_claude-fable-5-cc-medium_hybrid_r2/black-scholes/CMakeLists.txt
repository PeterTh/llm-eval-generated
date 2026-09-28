cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the cluster's GPUs (Ampere) unless overridden
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and OpenMP in the host compiler
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP runtime, CUDA runtime, and math library
target_link_libraries(black_scholes PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
