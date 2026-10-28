cmake_minimum_required(VERSION 3.24)

# Target the GPUs present on the build machine unless overridden. This must be set
# before project(), which otherwise pre-populates a default architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_EXTENSIONS ON)

# Required parallelization stack: MPI (across nodes/GPUs), OpenMP (host threads), CUDA (GPU kernels).
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The CUDA kernels live in cholesky.cpp, so it is compiled by nvcc (host code is forwarded to the C++ compiler).
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Use the same floating-point contraction policy as the original build (GNU dialect, FMA
# contraction enabled with -march=native) for both host and device code.
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp,-ffp-contract=fast>
)

target_link_libraries(cholesky PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m
)
