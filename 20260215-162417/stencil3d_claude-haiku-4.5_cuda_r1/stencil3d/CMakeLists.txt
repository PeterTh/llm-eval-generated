cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA toolkit path
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda/include)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

# Set C++ as CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/c++)

add_executable(stencil3d stencil3d.cpp)

# Set the language to CUDA for this file
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library and CUDA runtime
target_link_libraries(stencil3d m)
