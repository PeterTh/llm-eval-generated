cmake_minimum_required(VERSION 3.24)
# Select the installed GPU unless the caller explicitly supplies targets.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

# Compile the existing source as CUDA without changing its filename.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
# Do not contract/reassociate double arithmetic: result hashes depend on the
# original per-neighbor rounding and summation order.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
target_link_libraries(unstructured PRIVATE m)
