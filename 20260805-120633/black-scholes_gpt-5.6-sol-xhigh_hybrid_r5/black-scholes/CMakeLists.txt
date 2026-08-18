cmake_minimum_required(VERSION 3.24)

# Select the build node's accelerator by default. This must precede project(),
# because CUDA otherwise initializes the variable to its legacy compiler
# default. Cluster builds can supply a fat-binary list with
# -DCMAKE_CUDA_ARCHITECTURES=... instead.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Retain the existing filename while compiling it as CUDA. Callers can override
# CMAKE_CUDA_ARCHITECTURES with a cluster-wide fat-binary architecture list.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --extra-device-vectorization
        --restrict
        --allow-unsupported-compiler
        -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra
    >
)

target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
