cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Treat matmul.cpp as CUDA source so it is compiled with NVCC
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable separable compilation and set CUDA standard
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)

# Compiler options for host C++ and CUDA
target_compile_options(matmul PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
)

# Link math library
target_link_libraries(matmul PRIVATE m)
