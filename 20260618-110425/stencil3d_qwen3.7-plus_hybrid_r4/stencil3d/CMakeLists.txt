cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architecture for RTX 3090 (compute 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Treat .cpp as CUDA source
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -Wall -Wextra)

# Pass OpenMP flags to CUDA host compiler
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp>)

# Link libraries
target_link_libraries(stencil3d MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(stencil3d m)


