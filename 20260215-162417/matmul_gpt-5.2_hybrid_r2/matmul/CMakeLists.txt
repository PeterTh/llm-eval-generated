cmake_minimum_required(VERSION 3.18)

# Help CMake find nvcc on accelerator nodes where it may not be on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(_NVCC nvcc HINTS /usr/local/cuda/bin /opt/cuda/bin)
  if(_NVCC)
    set(CMAKE_CUDA_COMPILER "${_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

# CMake can't auto-detect an architecture on systems without a visible GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Compile this TU with NVCC so we can keep a single source file.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
