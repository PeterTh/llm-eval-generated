cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-O3 -arch=sm_60 -std=c++20")

cuda_add_executable(unstructured unstructured.cpp)
target_compile_options(unstructured PRIVATE -O3 -Wall -Wextra)
target_link_libraries(unstructured m)
