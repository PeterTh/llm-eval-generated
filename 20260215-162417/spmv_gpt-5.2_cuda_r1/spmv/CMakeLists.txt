cmake_minimum_required(VERSION 3.10)

# nvcc is available in this environment but may not be on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build a portable fatbinary across major NVIDIA GPU architectures.
set(CMAKE_CUDA_ARCHITECTURES all-major)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(spmv m)
