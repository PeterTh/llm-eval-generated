cmake_minimum_required(VERSION 3.10)

# Prefer a CUDA toolkit in standard locations even if nvcc isn't on PATH
if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# Compile the existing .cpp as CUDA (no file renames).
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -lineinfo -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-fopenmp>
)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
