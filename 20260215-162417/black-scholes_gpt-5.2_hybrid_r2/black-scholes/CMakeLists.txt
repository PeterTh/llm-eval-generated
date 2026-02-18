cmake_minimum_required(VERSION 3.10)

# Ensure nvcc is found in environments where it's installed but not on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Avoid relying on runtime GPU detection during configuration.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Compile the existing source file with NVCC (hybrid CUDA+host code)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

# Link libraries
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
