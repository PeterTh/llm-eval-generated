cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU targets (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# MPI for domain decomposition, OpenMP for host-side parallel work, CUDA for the stencil kernels
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The program is a single CUDA translation unit (host code plus kernels)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations. --fmad=false keeps device arithmetic to the explicit fma() calls,
# so results match the reference build bit for bit.
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

# Link math library, MPI, OpenMP and the CUDA runtime
target_link_libraries(cahn_hilliard m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
