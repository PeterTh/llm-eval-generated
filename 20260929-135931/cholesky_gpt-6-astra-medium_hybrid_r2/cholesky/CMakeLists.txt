cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit 11.0 REQUIRED)

add_executable(cholesky cholesky.cpp)
target_compile_features(cholesky PRIVATE cxx_std_17)
target_compile_options(cholesky PRIVATE -O3 -Wall -Wextra)
# CUDA's own BLAS/solver libraries supply the GPU kernels; no .cu file is needed.
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
    CUDA::cudart CUDA::cublas CUDA::cusolver)
