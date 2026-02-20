cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Enable CUDA
enable_language(CUDA)

add_executable(stencil3d stencil3d.cu)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(stencil3d m)
