cmake_minimum_required(VERSION 3.18)

# Ensure a CUDA compiler is found even when nvcc isn't on PATH in the benchmark environment.
if(EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()

# Provide a reasonable default architecture for accelerator clusters (override via -DCMAKE_CUDA_ARCHITECTURES=...).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "" FORCE)
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Compile the .cpp file with NVCC so we can unconditionally use CUDA.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link libraries
target_link_libraries(cholesky PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cublas CUDA::cusolver)

# Reasonable defaults for CUDA compilation
set_target_properties(cholesky PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
