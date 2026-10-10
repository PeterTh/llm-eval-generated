cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# cuBLAS is supplied by the CUDA toolkit; every build uses MPI, OpenMP and CUDA.
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                    CUDA::cudart CUDA::cublas m)
