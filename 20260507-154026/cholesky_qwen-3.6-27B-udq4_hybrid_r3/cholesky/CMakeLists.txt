cmake_minimum_required(VERSION 3.17)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, OpenMP, CUDA, cuBLAS, cuSOLVER, and math
target_link_libraries(cholesky
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    cublas
    cusolver
    m)
