cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compile the .cpp file as CUDA (it contains __global__ kernels)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations for host compiler via nvcc
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

# Include directories for MPI and OpenMP
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(cahn_hilliard
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    m
)

# Set CUDA architectures
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)
