cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures - native will auto-detect, or fall back to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(nbody nbody.cu)

# Enable optimizations for CUDA
set_target_properties(nbody PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(nbody PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(nbody m)
