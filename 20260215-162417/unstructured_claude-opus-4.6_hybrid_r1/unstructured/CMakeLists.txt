cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile .cpp file as CUDA so nvcc handles __global__ etc.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "70;80;86;90")

# CUDA compile options: -O3 for device, host flags via -Xcompiler
target_compile_options(unstructured PRIVATE
    -O3 -Xcompiler=-fopenmp,-O3,-march=native)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE ${MPI_CXX_LIBRARIES} m)
target_link_options(unstructured PRIVATE -fopenmp)
