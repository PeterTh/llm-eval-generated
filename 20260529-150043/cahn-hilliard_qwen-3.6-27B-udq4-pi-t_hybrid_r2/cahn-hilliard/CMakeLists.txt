cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP (for flag detection)
find_package(OpenMP COMPONENTS CXX)

# Create the executable; the .cpp source contains CUDA device code
# so we set it to be compiled with nvcc
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Mark the source file to be compiled with nvcc (CUDA language)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES
    LANGUAGE CUDA
)

# CUDA architecture for RTX 3090 (compute 8.6)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES 86
)

# Link MPI, CUDA runtime, and math
# OpenMP is handled via compile flags (-Xcompiler=-fopenmp)
target_link_libraries(cahn_hilliard
    PRIVATE
        MPI::MPI_CXX
        cudart
        m
)

# Include directories
target_include_directories(cahn_hilliard PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/..
)

# Compile options: O3, native arch, OpenMP via -Xcompiler for nvcc
target_compile_options(cahn_hilliard PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-fopenmp
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link OpenMP runtime for host code pragmas
if(OpenMP_CXX_FOUND)
    target_link_options(cahn_hilliard PRIVATE ${OpenMP_CXX_FLAGS})
endif()
