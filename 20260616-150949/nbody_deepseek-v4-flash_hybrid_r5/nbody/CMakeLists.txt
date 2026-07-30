cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Mark source as CUDA so nvcc compiles it (handles __global__ kernels)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations and OpenMP (must pass through nvcc -Xcompiler for host side)
target_compile_options(nbody PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>
)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(nbody PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

# Set CUDA architecture for NVIDIA GPU code generation
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "86"
)
