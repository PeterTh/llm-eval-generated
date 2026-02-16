cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

find_package(CUDA REQUIRED)

cuda_add_executable(unstructured unstructured.cu)
target_link_libraries(unstructured m)

# Enable optimizations
