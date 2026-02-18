cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for broader compatibility (compute capability 6.0+)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)

# Compile .cpp file with CUDA compiler
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations for C++
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Enable optimizations for CUDA
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)

# Link math library
target_link_libraries(spmv m)
