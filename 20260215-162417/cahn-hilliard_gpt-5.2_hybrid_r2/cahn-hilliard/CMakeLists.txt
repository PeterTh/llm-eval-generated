cmake_minimum_required(VERSION 3.18)

# Point CMake at nvcc when CUDA isn't on PATH (common on stripped-down environments).
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()

# Ensure CUDA builds even on login nodes without a visible GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70-real;70-virtual;80-real;80-virtual CACHE STRING "" FORCE)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cahn_hilliard PRIVATE CUDA::cudart)

# Prefer building for the local GPU when supported by CMake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()
