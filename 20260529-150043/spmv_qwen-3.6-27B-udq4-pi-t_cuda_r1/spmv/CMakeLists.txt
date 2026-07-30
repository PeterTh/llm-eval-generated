cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

enable_language(CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture for RTX 3090 (compute 8.6) + broader compatibility
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-march=native,-Wall,-Wextra")

# Link math library
target_link_libraries(spmv m)
