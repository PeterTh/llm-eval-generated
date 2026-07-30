cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Force compilation with CUDA compiler
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (sm_52 targets Maxwell and later GPUs with FP64 support)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "52")
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(matmul m)
