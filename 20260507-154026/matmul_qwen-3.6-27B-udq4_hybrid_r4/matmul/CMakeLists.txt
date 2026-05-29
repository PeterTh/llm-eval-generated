cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Generate CUDA kernel source in the build directory
set(KERNEL_CU "${CMAKE_CURRENT_BINARY_DIR}/matmul_kernel.cu")
set(KERNEL_H "${CMAKE_CURRENT_BINARY_DIR}/matmul_kernel.h")

file(WRITE "${KERNEL_H}"
"#pragma once\n"
"void gpuMatmul(const double* A, const double* B, double* C, int localRows, int N);\n"
)

file(WRITE "${KERNEL_CU}"
"#include \"cuda_runtime.h\"\n"
"#include \"device_launch_parameters.h\"\n"
"\n"
"template <int TILE_DIM>\n"
"__global__ void matmulKernel(const double* __restrict__ A,\n"
"                             const double* __restrict__ B,\n"
"                             double* __restrict__ C,\n"
"                             const int M, const int N, const int K) {\n"
"    __shared__ double sA[TILE_DIM][TILE_DIM];\n"
"    __shared__ double sB[TILE_DIM][TILE_DIM];\n"
"    const int row = blockIdx.y * TILE_DIM + threadIdx.y;\n"
"    const int col = blockIdx.x * TILE_DIM + threadIdx.x;\n"
"    double sum = 0.0;\n"
"    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {\n"
"        const int aCol = t * TILE_DIM + threadIdx.x;\n"
"        const int bRow = t * TILE_DIM + threadIdx.y;\n"
"        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;\n"
"        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE_DIM; ++k) {\n"
"            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < M && col < N) {\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"void gpuMatmul(const double* A, const double* B, double* C,\n"
"               int localRows, int N) {\n"
"    const int TILE = 32;\n"
"    dim3 blockDim(TILE, TILE);\n"
"    dim3 gridDim((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);\n"
"    double *dA = nullptr, *dB = nullptr, *dC = nullptr;\n"
"    cudaMalloc(&dA, N * N * sizeof(double));\n"
"    cudaMalloc(&dB, N * N * sizeof(double));\n"
"    cudaMalloc(&dC, localRows * N * sizeof(double));\n"
"    cudaMemcpy(dA, A, N * N * sizeof(double), cudaMemcpyHostToDevice);\n"
"    cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice);\n"
"    matmulKernel<32><<<gridDim, blockDim>>>(dA, dB, dC, localRows, N, N);\n"
"    cudaDeviceSynchronize();\n"
"    cudaMemcpy(C, dC, localRows * N * sizeof(double), cudaMemcpyDeviceToHost);\n"
"    cudaFree(dA);\n"
"    cudaFree(dB);\n"
"    cudaFree(dC);\n"
"}\n"
)

# Compile kernel with nvcc
enable_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE STRING "CUDA host compiler" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

add_library(cuda_kernel OBJECT "${KERNEL_CU}")
set_target_properties(cuda_kernel PROPERTIES
    CUDA_ARCHITECTURES "86"
)
target_compile_options(cuda_kernel PRIVATE -O3 -std=c++20)
target_include_directories(cuda_kernel PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# Host executable (compiled with mpicxx for MPI + OpenMP)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
target_include_directories(matmul PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(matmul MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Link kernel object into executable
target_sources(matmul PRIVATE $<TARGET_OBJECTS:cuda_kernel>)
