cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Enable CUDA separate compilation for larger kernels
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Compile spmv.cpp with nvcc so CUDA kernel syntax is supported
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")

# Link MPI, OpenMP, and math library
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
