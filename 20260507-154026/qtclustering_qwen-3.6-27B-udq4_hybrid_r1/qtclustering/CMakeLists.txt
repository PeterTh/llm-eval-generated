cmake_minimum_required(VERSION 3.10)

project(qtclustering CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Locate CUDA runtime
find_library(CUDA_CUDART_LIBRARY cudart PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
find_path(CUDA_INCLUDE_DIR cuda_runtime.h PATHS /usr/local/cuda/include /usr/include)
find_program(NVCC_EXECUTABLE nvcc)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(qtclustering qtclustering.cpp)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES LINKER_LANGUAGE CXX)

target_compile_options(qtclustering PRIVATE
    --std=c++20 -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-fopenmp
)

target_include_directories(qtclustering PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIR}
)

target_link_libraries(qtclustering
    MPI::MPI_CXX
    ${CUDA_CUDART_LIBRARY}
    m
)

# Add OpenMP link flags
target_link_options(qtclustering PRIVATE -fopenmp)
