cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA RTX 3090 (sm_86) and later architectures
set(CMAKE_CUDA_ARCHITECTURES "86")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile as CUDA source to support __global__ kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# nvcc flags: -O3 for device, -Xcompiler passes flags to host compiler (gcc)
target_compile_options(qtclustering PRIVATE
    -O3
    "-Xcompiler=-march=native,-Wall,-Wextra,-fopenmp"
)

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
