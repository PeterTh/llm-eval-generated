cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) present in this machine by default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark is compiled as CUDA so it unconditionally runs on the GPU.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations. Floating-point contraction (FMA) is disabled to keep
# the arithmetic bit-for-bit identical to the reference implementation.
target_compile_options(unstructured PRIVATE
    -O3
    --fmad=false
    -Xptxas=-O3
    -Xcompiler=-march=native,-Wall,-Wextra)

# Link math library (CUDA runtime is linked automatically)
target_link_libraries(unstructured m)
