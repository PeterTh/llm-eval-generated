cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Compile .cpp file with CUDA compiler
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Target GPU architecture (RTX 3090 = sm_86)
set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(stencil3d m)
