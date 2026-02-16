cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Treat source as CUDA so NVCC compiles it
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Add CUDA-specific compile options
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library and CUDA runtime
target_link_libraries(spmv m cudart)
