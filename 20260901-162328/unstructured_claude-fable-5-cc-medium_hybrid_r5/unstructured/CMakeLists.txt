cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# The single translation unit contains CUDA kernels: compile it with nvcc
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host compiler flags forwarded through nvcc)
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

# MPI include dirs/libs added directly to avoid forwarding host-only compile
# flags (e.g. -pthread) to nvcc
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE ${MPI_CXX_LIBRARIES}
  OpenMP::OpenMP_CXX m)
