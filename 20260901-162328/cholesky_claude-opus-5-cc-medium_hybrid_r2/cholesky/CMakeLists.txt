cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The benchmark contains CUDA kernels; compile it with the CUDA compiler.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations (host flags are forwarded through nvcc)
target_compile_options(cholesky PRIVATE
  -O3
  -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra
)

target_link_libraries(cholesky PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cublas
  CUDA::cudart
  m
)
