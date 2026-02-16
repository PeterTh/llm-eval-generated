cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp as a CUDA source and enable separable compilation
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
# CUDA-specific compile options
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math> )

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(spmv m)
