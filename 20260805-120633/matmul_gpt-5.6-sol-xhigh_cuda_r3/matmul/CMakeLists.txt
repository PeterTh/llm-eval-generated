cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Keep the existing filename while compiling it as a CUDA translation unit.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--compiler-options=-O3,-march=native,-Wall,-Wextra>"
)
set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# cuBLAS is part of the CUDA toolkit and provides the vendor-tuned DGEMM kernel.
target_link_libraries(matmul PRIVATE CUDA::cublas)
