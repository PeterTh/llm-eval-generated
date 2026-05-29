cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Compile matmul.cpp with nvcc (it contains CUDA kernel code)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations – nvcc needs -Xcompiler to forward flags to host compiler
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-march=native,-fopenmp,-Wall,-Wextra)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "86")

# Link MPI, OpenMP, CUDA runtime and math
target_link_libraries(matmul
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cuda
    m)
