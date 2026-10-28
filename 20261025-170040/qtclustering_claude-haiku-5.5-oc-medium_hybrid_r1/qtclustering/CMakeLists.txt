cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Required parallel runtimes: MPI (distributed), OpenMP (host threads), CUDA (GPU)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# The source contains device kernels, so it is compiled by nvcc. Host code is
# passed to the host compiler, which must use the same floating-point contraction
# rules as the original build (-ffp-contract=fast) to keep results identical.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(qtclustering PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m
)
