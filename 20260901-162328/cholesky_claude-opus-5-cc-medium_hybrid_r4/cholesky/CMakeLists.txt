cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(CMAKE_CUDA_ARCHITECTURES native)
else()
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The source contains CUDA kernels and is compiled with nvcc.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations (host compiler flags are forwarded through nvcc)
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(cholesky
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cublas
  CUDA::cudart
  m
)
