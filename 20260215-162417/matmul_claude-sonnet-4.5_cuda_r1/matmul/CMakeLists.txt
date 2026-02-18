cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project declaration
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cu)

# Enable optimizations
target_compile_options(matmul PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

set_target_properties(matmul PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Link math library
target_link_libraries(matmul m)
