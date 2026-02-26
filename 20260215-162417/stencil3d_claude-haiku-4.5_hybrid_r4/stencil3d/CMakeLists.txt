cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Link MPI, OpenMP, and CUDA
target_link_libraries(stencil3d PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    m
)

# Include directories
target_include_directories(stencil3d PRIVATE ${CUDA_INCLUDE_DIRS})

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA compilation flags
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -arch=sm_70
    --relocatable-device-code=true
>)
