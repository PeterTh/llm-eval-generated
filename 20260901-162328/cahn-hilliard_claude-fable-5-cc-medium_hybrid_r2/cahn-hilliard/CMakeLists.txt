cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Single-source hybrid MPI+OpenMP+CUDA implementation: compile the .cpp as CUDA
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link with the host C++ compiler so OpenMP/MPI link flags are handled natively
set_target_properties(cahn_hilliard PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(cahn_hilliard PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m)
