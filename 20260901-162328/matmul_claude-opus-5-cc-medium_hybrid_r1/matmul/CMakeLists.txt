cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# matmul.cpp is compiled twice: nvcc builds the device code (the part guarded by
# __CUDACC__) via a symbolic link with a .cu extension, while the host compiler
# builds everything else. The host compiler generates noticeably faster code for
# the CPU part of the hybrid GEMM than nvcc's host pass does.
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp ${CMAKE_CURRENT_BINARY_DIR}/matmul_device.cu
     SYMBOLIC)

add_library(matmul_device OBJECT ${CMAKE_CURRENT_BINARY_DIR}/matmul_device.cu)
target_include_directories(matmul_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(matmul_device PRIVATE -O3)
target_link_libraries(matmul_device PRIVATE MPI::MPI_CXX)

add_executable(matmul matmul.cpp $<TARGET_OBJECTS:matmul_device>)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
