cmake_minimum_required(VERSION 3.10)

# Ensure CUDA arch is always set for CUDA language enablement.
# "native" yields best performance on the build machine/cluster nodes.
set(CMAKE_CUDA_ARCHITECTURES native)

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile the single-source benchmark with NVCC to unconditionally use CUDA
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
# NOTE: disable CUDA FMAD to better preserve CPU-like floating-point semantics for hashing.
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false --compiler-options=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
