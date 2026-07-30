cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# CUDA compile options
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Target CUDA architecture for RTX 3090 (sm_86)
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "86")
