cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU code generation targets (Volta through Hopper), unless overridden.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Required parallelization stack: MPI (ranks), OpenMP (host threads), CUDA (GPU kernels).
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The source contains CUDA kernels, so it is compiled by nvcc with the host
# compiler handling the C++ parts.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Optimized host and device code. Floating-point contraction (FMA) is disabled on both
# sides so every cell result follows the source expression exactly, independent of
# decomposition and of whether the code runs on host or device.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-ffp-contract=off,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(cahn_hilliard PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
