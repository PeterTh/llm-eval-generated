cmake_minimum_required(VERSION 3.18)

# Avoid CUDA compiler / arch auto-detection failures on headless build nodes.
# Override with -DCMAKE_CUDA_COMPILER=... and/or -DCMAKE_CUDA_ARCHITECTURES=... as needed.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Compile the existing .cpp translation unit with NVCC (no new files, same executable name)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
