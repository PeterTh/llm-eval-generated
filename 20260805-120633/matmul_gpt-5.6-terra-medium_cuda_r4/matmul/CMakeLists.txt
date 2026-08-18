cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Build the executable as optimized CUDA code.
set_target_properties(matmul PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                     CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(matmul m)
