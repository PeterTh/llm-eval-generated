cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# RTX 3090 compute capability
set(CMAKE_CUDA_ARCHITECTURES "86")

# Set per-file CUDA compilation: nvcc will compile this as CUDA source.
# We pass host compiler flags through -Xcompiler to nvcc for the host compiler.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra")

add_executable(unstructured unstructured.cpp)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
