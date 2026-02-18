cmake_minimum_required(VERSION 3.18)

# Ensure CMake can locate NVCC in environments where it's not on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Required on some systems where CMake cannot infer a default GPU arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile this TU with NVCC so we can embed CUDA kernels without adding new files.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET roomsim PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic -Xcompiler=-fopenmp>
)

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
