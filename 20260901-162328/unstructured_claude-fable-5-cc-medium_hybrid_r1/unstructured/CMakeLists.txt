cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPUs present on the build machine unless overridden
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations. --fmad=false keeps device arithmetic bit-identical to
# the serial reference (no FMA contraction in the flux accumulation).
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

# Link MPI, OpenMP runtime, CUDA runtime, and math library
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
