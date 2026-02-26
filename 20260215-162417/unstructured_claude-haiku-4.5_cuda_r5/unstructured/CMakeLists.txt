cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Suppress CMP0104 policy warning
cmake_policy(SET CMP0104 OLD)

# Add CUDA flags for better performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60")

add_executable(unstructured unstructured.cpp)

# Enable optimizations for C++
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Set the source file to be compiled with CUDA compiler
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Link math library
target_link_libraries(unstructured m)
