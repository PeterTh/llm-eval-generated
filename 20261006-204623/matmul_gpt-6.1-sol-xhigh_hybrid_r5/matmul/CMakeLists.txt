cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All three components are mandatory; cuBLAS ships with the CUDA toolkit.
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
# Use only the MPI C interface, avoiding legacy C++ binding headers.
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                    CUDA::cudart CUDA::cublas m)
