cmake_minimum_required(VERSION 3.24)

# Target the GPUs present on the build machine for maximum performance.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The benchmark is implemented with CUDA; compile the existing source as a
# CUDA translation unit so the kernels and host code stay in one file.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

find_package(CUDAToolkit REQUIRED)

set_target_properties(spmv PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# Link CUDA runtime and math library
target_link_libraries(spmv PRIVATE CUDA::cudart m)
