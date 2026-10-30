cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m cudart)
