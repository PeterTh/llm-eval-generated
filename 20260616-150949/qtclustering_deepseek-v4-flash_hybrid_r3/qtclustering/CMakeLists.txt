cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile qtclustering.cpp with CUDA (nvcc) to support GPU kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
    -O3
    -arch=sm_86
    -Xcompiler=-Wall,-Wextra,-march=native,-fopenmp
    --use_fast_math
)

target_link_libraries(qtclustering
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
