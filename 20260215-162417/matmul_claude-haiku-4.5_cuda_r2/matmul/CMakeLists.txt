cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA toolkit path
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6)

# Enable CUDA for C++ files
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Include CUDA headers
target_include_directories(matmul PRIVATE /usr/local/cuda-12.6/include)

# Link math library
target_link_libraries(matmul m)
