cmake_minimum_required(VERSION 3.18)

# nvcc needs a host compiler it supports; fall back to g++-12 if the default
# g++ is too new for the installed CUDA toolkit (e.g. CUDA 12.0 caps at GCC 12)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
  execute_process(COMMAND g++ -dumpversion OUTPUT_VARIABLE _GXX_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_GXX_VER VERSION_GREATER 12 AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
  endif()
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)
target_link_options(unstructured PRIVATE -fopenmp)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
