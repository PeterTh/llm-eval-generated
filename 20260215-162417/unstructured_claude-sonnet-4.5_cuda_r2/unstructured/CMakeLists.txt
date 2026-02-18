cmake_minimum_required(VERSION 3.18)

# Set CUDA architecture before project (RTX 3090 = 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cu)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link math library
target_link_libraries(unstructured m CUDA::cudart)
