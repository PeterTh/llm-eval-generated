cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Compile the source as CUDA so its kernels and __host__ __device__ code build with nvcc
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall,-Wextra>
)

find_package(CUDAToolkit REQUIRED)

# Link math and CUDA runtime/curand libraries
target_link_libraries(roomsim m CUDA::cudart CUDA::curand)
