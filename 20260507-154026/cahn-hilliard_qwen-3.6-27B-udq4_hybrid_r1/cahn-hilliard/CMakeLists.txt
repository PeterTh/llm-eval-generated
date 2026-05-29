cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# Determine CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()

add_executable(cahn_hilliard cahn_hilliard.cpp)

set_target_properties(cahn_hilliard PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fopenmp>
)

# Link MPI, OpenMP, math, and CUDA runtime
target_link_libraries(cahn_hilliard
    MPI::MPI_CXX
    m
    $<$<CXX_COMPILER_ID:GNU>:omp>
    $<$<CXX_COMPILER_ID:Clang>:omp>
)

# Include directories for MPI
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_PATH})
