cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70 -use_fast_math>)
else()
    target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link math library
target_link_libraries(matmul m)
