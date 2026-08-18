cmake_minimum_required(VERSION 3.18)

# Select the native benchmark GPU unless the caller supplied an architecture.
# This must be set before project() enables CUDA so CMake does not populate its
# default architecture first.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing .cpp filename while compiling the benchmark with nvcc.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations for both the host portions and the CUDA kernel.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-march=native,-Wall,-Wextra>
)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Link math library
target_link_libraries(unstructured PRIVATE m)
