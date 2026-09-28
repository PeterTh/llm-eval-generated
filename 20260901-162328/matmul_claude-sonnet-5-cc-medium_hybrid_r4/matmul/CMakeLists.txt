cmake_minimum_required(VERSION 3.17)
project(matmul_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP and CUDA (cuBLAS) libraries
target_link_libraries(matmul PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
