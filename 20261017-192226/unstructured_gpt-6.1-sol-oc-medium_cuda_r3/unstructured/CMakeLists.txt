cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Keep separate multiply/add operations: the result hash includes every FP bit.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(unstructured m)
