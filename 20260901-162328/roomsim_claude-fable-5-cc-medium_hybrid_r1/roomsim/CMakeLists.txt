cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on the build machine by default
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(roomsim roomsim.cpp)

# The single source file contains CUDA kernels: compile it with nvcc
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (device and host side) and OpenMP in host code
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)
target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX)

# Link with the host C++ toolchain
set_target_properties(roomsim PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(roomsim PRIVATE
  CUDA::cudart
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  m)
