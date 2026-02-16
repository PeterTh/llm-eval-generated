cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Find MPI and OpenMP for hybrid parallelism
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA support is optional in this environment; build CPU-only MPI+OpenMP by default
message(STATUS "Building without CUDA support; USE_CUDA=0")
target_compile_definitions(unstructured PRIVATE USE_CUDA=0)

# Link MPI and OpenMP and math library
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
