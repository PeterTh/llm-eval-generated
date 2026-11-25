cmake_minimum_required(VERSION 3.20)
project(spmv_benchmark LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA while keeping its filename.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
target_compile_options(spmv PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)
