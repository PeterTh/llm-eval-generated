cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Generate code for the GPU present at configure time. This avoids carrying
# unnecessary compatibility code and lets nvcc tune for all native features.
set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES native)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;--extra-device-vectorization;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)
