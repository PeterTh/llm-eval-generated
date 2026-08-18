cmake_minimum_required(VERSION 3.18)

# The benchmark is intentionally CUDA-only: its update loop has one GPU thread
# per mesh element and does not provide a serial fallback.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Enable optimizations. Disable contraction/fast-math so the per-element
# floating-point operation order remains equivalent to the reference code.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(unstructured PRIVATE CUDA::cudart m)
