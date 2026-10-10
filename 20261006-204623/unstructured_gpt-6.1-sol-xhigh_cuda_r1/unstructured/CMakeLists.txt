cmake_minimum_required(VERSION 3.24)

# Optimize for installed GPUs unless the caller selects deployment architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Preserve the reference arithmetic order; fast math and FMA contraction can
# change the bitwise result hashes even when values are numerically close.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)
target_link_libraries(unstructured PRIVATE m)
