cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Force CUDA compilation for this source file
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture explicitly for RTX 3090 (compute capability 8.6)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "86")

# Enable aggressive optimizations
# CUDA: restrict pointers, max optimization, use fast math
# CXX: auto-vectorization, native arch tuning
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--restrict -O3 --use_fast_math --ftz=true --prec-div=false --prec-sqrt=false>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -mtune=native -Wall -Wextra>)

# Pass host compiler optimizations through nvcc
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3 -Xcompiler=-march=native>)

# Link math library
target_link_libraries(unstructured m)
