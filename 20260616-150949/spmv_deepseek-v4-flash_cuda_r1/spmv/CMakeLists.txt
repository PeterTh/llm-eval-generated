cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the available GPU architecture (NVIDIA RTX 3090, SM 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

# Link math library
target_link_libraries(spmv m)
