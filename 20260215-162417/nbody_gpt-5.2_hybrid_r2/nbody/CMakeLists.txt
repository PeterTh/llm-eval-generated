cmake_minimum_required(VERSION 3.18)

# Ensure NVCC is found even if /usr/local/cuda/bin is not on PATH.
if (EXISTS "/usr/local/cuda/bin/nvcc" AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Avoid configure-time GPU probing on headless systems.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(nbody nbody.cpp)

# Compile the existing .cpp as CUDA to allow host+device code in one file.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -lineinfo>
)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
