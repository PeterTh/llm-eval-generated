cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Rename source to .cu for nvcc compilation
add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link cuBLAS
find_library(CUBLAS_LIBRARY cublas)
find_library(CUDART_LIBRARY cudart)
target_link_libraries(cholesky ${CUBLAS_LIBRARY} ${CUDART_LIBRARY} m)
