cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use nvcc from the local CUDA install (often not on PATH in eval environments).
if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

# CMake may fail to auto-detect an architecture in headless environments; build a fatbin.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 CACHE STRING "CUDA architectures")
endif()

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Compile this translation unit with NVCC while keeping the existing filename.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(cholesky m)
