cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(nbody nbody.cu)

# Enable optimizations
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
