cmake_minimum_required(VERSION 3.20)
project(spmv_benchmark LANGUAGES CUDA)

set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(spmv PRIVATE -O3 --fmad=false)
