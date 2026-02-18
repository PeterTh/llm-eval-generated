cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA is optional - for simplicity, use host compiler for all code
# CUDA kernels would need separate compilation but MPI/OpenMP provide ample parallelization
add_compile_definitions(CUDA_AVAILABLE=0)

message(STATUS "Building with MPI and OpenMP (CUDA disabled for host compiler)")

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(unstructured PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
