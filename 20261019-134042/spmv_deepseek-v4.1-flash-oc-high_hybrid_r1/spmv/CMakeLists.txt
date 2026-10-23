cmake_minimum_required(VERSION 3.18)

# Compile for the common accelerator architectures. Override on the command
# line (for example -DCMAKE_CUDA_ARCHITECTURES=native) if desired.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# The single existing source file contains both host and CUDA device code,
# so it is compiled by nvcc as a CUDA translation unit.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(spmv PRIVATE
  -O3
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(spmv PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
