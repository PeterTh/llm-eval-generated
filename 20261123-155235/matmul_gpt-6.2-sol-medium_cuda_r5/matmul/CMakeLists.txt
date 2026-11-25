cmake_minimum_required(VERSION 3.20)
project(matmul_benchmark LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

target_compile_options(matmul PRIVATE -O3 -Xcompiler=-Wall,-Wextra)
