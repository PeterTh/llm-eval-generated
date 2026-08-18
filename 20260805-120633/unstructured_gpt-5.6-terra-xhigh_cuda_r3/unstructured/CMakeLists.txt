cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# The benchmark always executes on the GPU and is tuned for the Ampere GPU
# available to the benchmark environment.
set_target_properties(unstructured PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION OFF
)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Keep host-side setup optimized as well as the CUDA kernels.  Fast-math is
# intentionally not enabled: the simulation's double-precision update order
# is part of its numerical semantics.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

target_link_libraries(unstructured PRIVATE CUDA::cudart m)
