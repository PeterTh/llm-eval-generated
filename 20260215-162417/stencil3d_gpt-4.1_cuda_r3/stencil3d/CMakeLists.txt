cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> -O3 -Wall -Wextra)

# Link math library
target_link_libraries(stencil3d m)
