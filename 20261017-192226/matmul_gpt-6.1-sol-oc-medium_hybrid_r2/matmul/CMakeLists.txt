cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# CUDA's runtime and cuBLAS APIs are callable directly from C++; no .cu file
# or optional CPU-only implementation is needed.
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep IEEE double precision (in particular, do not enable fast-math).
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                    CUDA::cudart CUDA::cublas m)
