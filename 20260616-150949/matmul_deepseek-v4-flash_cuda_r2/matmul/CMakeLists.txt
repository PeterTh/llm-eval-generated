cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Force CUDA compilation of the source file
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture for RTX 3090 (Ampere)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "86")

# Enable host compiler optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link math library
target_link_libraries(matmul m)
