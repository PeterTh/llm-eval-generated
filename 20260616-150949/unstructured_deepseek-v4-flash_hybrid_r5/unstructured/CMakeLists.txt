cmake_minimum_required(VERSION 3.11)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-fopenmp -Xcompiler=-Wall -Xcompiler=-Wextra>
)

target_link_libraries(unstructured PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

target_include_directories(unstructured PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
