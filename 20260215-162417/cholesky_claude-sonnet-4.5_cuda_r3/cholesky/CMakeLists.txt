cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename source file to .cu
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --generate-line-info>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Set CUDA architectures (adjust based on your GPU)
set_property(TARGET cholesky PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)

# Link math library
target_link_libraries(cholesky m)
