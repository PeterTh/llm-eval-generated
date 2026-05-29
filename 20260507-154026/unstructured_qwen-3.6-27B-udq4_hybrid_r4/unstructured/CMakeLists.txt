cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

# Link MPI, OpenMP and math library
target_link_libraries(unstructured MPI::MPI_CXX OpenMP::OpenMP_CXX m)
