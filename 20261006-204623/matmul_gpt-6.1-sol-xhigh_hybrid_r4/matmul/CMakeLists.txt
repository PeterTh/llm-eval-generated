cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# CUDA's bundled cuBLAS supplies the optimized double-precision GPU GEMM.
# All three parallel runtimes are required; there is no serial/CPU fallback.
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                    CUDA::cudart CUDA::cublas)
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
endif()
