cmake_minimum_required(VERSION 3.18)

# GPU architectures to build for (A100, RTX 30xx, H100); override with -DCMAKE_CUDA_ARCHITECTURES=...
# Must be set before project() so CMake does not substitute its compiler default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The source holds the CUDA kernel, so nvcc compiles it; host code is passed to the host compiler via -Xcompiler
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
  -O3
  -Xcompiler=-march=native
  -Xcompiler=-Wall
  -Xcompiler=-Wextra
  -Xcompiler=${OpenMP_CXX_FLAGS}
)

target_link_options(qtclustering PRIVATE ${OpenMP_CXX_FLAGS})

# MPI, CUDA runtime, and math library
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX CUDA::cudart m)
