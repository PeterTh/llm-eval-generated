cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU present on the build machine unless the user overrides it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark source is compiled as CUDA (nvcc) for GPU parallelism.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations. Host-side flags are forwarded to the host compiler via
# -Xcompiler; device code is built with optimizations and without fast-math so
# results stay numerically faithful to the reference formula.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra>)

# Link math library
target_link_libraries(black_scholes m)
