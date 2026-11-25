cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all)
endif()
project(matmul_benchmark LANGUAGES CUDA)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

target_compile_options(matmul PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)
