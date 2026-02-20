cmake_minimum_required(VERSION 3.10)

# On newer CMake, ensure CUDA architectures are set; default to native for best performance.
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Compile the existing .cpp with NVCC so we can embed CUDA kernels without adding new files.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# NOTE: OpenMP flags must be passed via -Xcompiler when compiling with NVCC.
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
