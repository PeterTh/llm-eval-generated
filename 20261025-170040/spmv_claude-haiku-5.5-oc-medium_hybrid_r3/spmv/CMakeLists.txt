cmake_minimum_required(VERSION 3.24)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPUs present on the build machine unless overridden
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# spmv.cpp contains CUDA kernels and is compiled by nvcc; host code is
# forwarded to the host compiler via -Xcompiler.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations. Floating-point contraction (FMA) is disabled on both
# host and device so every multiply and add rounds exactly as written; this
# keeps GPU results bit-identical to the CPU reference.
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffp-contract=off -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native,-ffp-contract=off,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(spmv PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
)
