cmake_minimum_required(VERSION 3.24)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Hybrid MPI + OpenMP + CUDA: all three are required
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The source is compiled as CUDA so the GPU kernels live in the same file
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

# Enable optimizations (host code is compiled through nvcc -Xcompiler)
target_compile_options(cholesky PRIVATE
  -O3
  -Xcompiler=-O3
  -Xcompiler=-march=native
  -Xcompiler=-Wall
  -Xcompiler=-Wextra
  -Xcompiler=${OpenMP_CXX_FLAGS}
)
# Link MPI, OpenMP runtime, CUDA runtime and math library
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
