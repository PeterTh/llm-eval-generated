cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

project(stencil3d_benchmark LANGUAGES CXX CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 60)
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)

# Link math library
target_link_libraries(stencil3d m)
