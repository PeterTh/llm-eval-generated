cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -Wall -Wextra)

# Link math library
# Remove problematic cuda_add_library and link only math
# CUDA linking handled by cuda_add_executable
# Only link math library
