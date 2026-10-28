cmake_minimum_required(VERSION 3.24)

# Device code is built for common datacenter/workstation GPUs unless the user
# overrides it (e.g. -DCMAKE_CUDA_ARCHITECTURES=90 for a single target)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# spmv.cpp contains CUDA kernels, so it is compiled by nvcc with the
# host compiler handling the C++ code.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations. -fmad=false keeps device arithmetic as separate
# multiply and add so that results are bitwise identical to the host reference.
target_compile_options(spmv PRIVATE
  -O3
  $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(spmv PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m
)
