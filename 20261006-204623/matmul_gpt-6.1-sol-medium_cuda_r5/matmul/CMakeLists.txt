cmake_minimum_required(VERSION 3.24)
# Generate native GPU code by default to avoid first-launch PTX compilation.
# Cross builds can override this with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without renaming it.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(matmul m)
