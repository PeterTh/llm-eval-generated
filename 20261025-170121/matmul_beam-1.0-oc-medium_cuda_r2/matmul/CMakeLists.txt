cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math and CUDA libraries
target_link_libraries(matmul m cublas cudart)
