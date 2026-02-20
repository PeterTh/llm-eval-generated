cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES native)

# Enable CUDA for the executable
enable_language(CUDA)

add_executable(matmul matmul.cpp)

# Set CUDA as the language for the source file
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Link math library and CUDA libraries
target_link_libraries(matmul m)
