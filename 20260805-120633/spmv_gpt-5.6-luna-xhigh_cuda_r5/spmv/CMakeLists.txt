cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# spmv.cpp is kept as an existing source file, but is compiled by nvcc so the
# executable always uses the CUDA implementation.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)

# Build for the benchmark GPU.  The generated device code remains runnable on
# the installed RTX 30-series device without a runtime CPU fallback.
set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES 86)

# Link math library
target_link_libraries(spmv PRIVATE m)
