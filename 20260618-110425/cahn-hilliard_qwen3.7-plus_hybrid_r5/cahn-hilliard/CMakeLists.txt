cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architecture for RTX 3090 (sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cahn_hilliard cahn_hilliard.cu)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Pass OpenMP flags through nvcc to host compiler
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=${OpenMP_CXX_FLAGS}
>)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(cahn_hilliard
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    m
)
