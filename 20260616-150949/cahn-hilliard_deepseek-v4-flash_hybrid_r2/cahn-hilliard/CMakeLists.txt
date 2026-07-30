cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile as CUDA source for GPU kernel support
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable target architecture optimizations
target_compile_options(cahn_hilliard PRIVATE
    -O3 -march=native
    # Forward OpenMP to nvcc host compiler for #pragma omp support
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Link libraries
target_link_libraries(cahn_hilliard
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "86"
)
