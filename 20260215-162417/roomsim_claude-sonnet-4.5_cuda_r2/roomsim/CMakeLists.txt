cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cu)

# Set CUDA architecture (detect automatically or use common architectures)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
endif()

set_target_properties(roomsim PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall>
)

# Link CUDA libraries
target_link_libraries(roomsim CUDA::cudart CUDA::curand m)
