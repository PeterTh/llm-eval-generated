cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cu)

# Enable optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Link CUDA runtime
target_link_libraries(stencil3d cudart)
