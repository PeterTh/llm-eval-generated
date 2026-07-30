cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for compute capability 8.6 (RTX 30xx series)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(nbody nbody.cu)

# Enable optimizations for both C++ and CUDA
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# Link math library
target_link_libraries(nbody m)
