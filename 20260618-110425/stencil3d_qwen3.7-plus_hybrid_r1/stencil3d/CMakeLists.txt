cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# Find OpenMP (for host-side parallelism)
find_package(OpenMP REQUIRED)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(stencil3d stencil3d.cu)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fopenmp,-march=native,-O3>
)

# Link MPI, OpenMP, CUDA runtime, math library
target_link_libraries(stencil3d
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    m
)
