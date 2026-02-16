cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Ensure spmv.cpp is compiled with NVCC
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host and device
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Find CUDA (older CMake compatibility)
find_package(CUDA REQUIRED)

# Link math library and CUDA runtime
target_link_libraries(spmv PRIVATE m ${CUDA_LIBRARIES})
