cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cu)

# Set CUDA architecture - using compute capability 7.0+ for modern GPUs
set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(unstructured m)
