cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
# cuBLAS and the CUDA runtime are supplied by the CUDA toolkit. The source
# uses the CUDA library API, so it does not require a separate .cu file.
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                     CUDA::cudart CUDA::cublas)
