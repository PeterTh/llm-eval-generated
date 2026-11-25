cmake_minimum_required(VERSION 3.20)
project(spmv_benchmark LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)
# Match the separate multiplication and addition used in the reference.
target_compile_options(spmv PRIVATE -O3 --fmad=false
    -Xcompiler=-O3,-ffp-contract=off,-Wall,-Wextra)
