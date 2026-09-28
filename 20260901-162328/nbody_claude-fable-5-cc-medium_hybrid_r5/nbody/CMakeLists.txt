cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-compiler flags passed through nvcc)
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math library, MPI, and OpenMP
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
