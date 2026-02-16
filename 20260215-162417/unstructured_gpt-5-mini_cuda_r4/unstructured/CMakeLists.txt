cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Treat the source as CUDA so nvcc compiles it (allows __global__ kernels in .cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler "-fPIC">)

# Link math and CUDA runtime libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(unstructured PRIVATE m CUDA::cudart)
