cmake_minimum_required(VERSION 3.18)

# Help CMake find nvcc in environments where CUDA isn't on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Compile this TU with nvcc so we can embed CUDA kernels without adding new files.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Prefer building for the local GPU for peak performance (can be overridden by users).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
