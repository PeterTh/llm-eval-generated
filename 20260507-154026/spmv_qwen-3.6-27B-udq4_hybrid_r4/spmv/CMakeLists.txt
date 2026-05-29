cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

add_executable(spmv spmv.cu)

# Enable optimizations with OpenMP and CUDA
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)
target_link_libraries(spmv MPI::MPI_CXX gomp m)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "86")
