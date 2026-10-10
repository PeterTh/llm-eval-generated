cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA target architectures")
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling it with nvcc.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Preserve the reference's double-precision operation order; contraction can
# otherwise change the result hashes even without --use_fast_math.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(unstructured m)
