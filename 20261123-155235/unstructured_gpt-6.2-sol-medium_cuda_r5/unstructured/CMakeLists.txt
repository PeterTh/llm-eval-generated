cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
target_compile_options(unstructured PRIVATE -O3 --fmad=false)
