cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Force CUDA compilation for this source file
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# CUDA architecture targeting Ampere and later GPUs for max performance
set_target_properties(stencil3d PROPERTIES
    CUDA_ARCHITECTURES "80;86"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(stencil3d m)
