cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture (compute capability 5.2+ covers Maxwell and later)
set(CMAKE_CUDA_ARCHITECTURES "52")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_executable(matmul matmul.cpp)

# Compile matmul.cpp with CUDA compiler
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable C++ host optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(matmul m)
