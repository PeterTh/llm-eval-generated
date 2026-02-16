cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language and treat .cpp as CUDA source so NVCC compiles it
enable_language(CUDA)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
# CUDA-specific optimizations
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Find MPI and OpenMP and CUDAToolkit
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Enable separable compilation for CUDA
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
