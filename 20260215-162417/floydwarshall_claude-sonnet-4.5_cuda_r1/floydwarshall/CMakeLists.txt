cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA path if needed
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
endif()

add_executable(floydwarshall floydwarshall.cu)

# Set CUDA architecture (compute capability) - use native for best performance
set_target_properties(floydwarshall PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(floydwarshall m)
