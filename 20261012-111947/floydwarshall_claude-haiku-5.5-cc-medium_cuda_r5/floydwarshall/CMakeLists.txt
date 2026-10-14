cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures: Volta through Hopper
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# The source is compiled by nvcc, which hosts the CUDA kernel
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall,-Wextra)

# Link CUDA runtime and math library
target_link_libraries(floydwarshall CUDA::cudart m)
