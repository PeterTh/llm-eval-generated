cmake_minimum_required(VERSION 3.20)
project(matmul_benchmark LANGUAGES CUDA)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(matmul PRIVATE -O3)
