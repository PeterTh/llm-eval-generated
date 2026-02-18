cmake_minimum_required(VERSION 3.17)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture to common values for compilation
set(CMAKE_CUDA_ARCHITECTURES "60;70;80;90")

# Enable CUDA language
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA file extensions to include .cpp
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# Link math library
target_link_libraries(spmv m)
