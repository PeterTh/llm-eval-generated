cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host code
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA architecture for RTX 3090 (compute 8.6) with broad compat
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "86")

# CUDA compiler optimizations
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Link math library
target_link_libraries(floydwarshall m)
