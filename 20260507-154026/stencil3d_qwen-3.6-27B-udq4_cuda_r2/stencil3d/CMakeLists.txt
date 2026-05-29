cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 30xx series
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Force nvcc to compile the .cpp file (it contains CUDA code)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)
