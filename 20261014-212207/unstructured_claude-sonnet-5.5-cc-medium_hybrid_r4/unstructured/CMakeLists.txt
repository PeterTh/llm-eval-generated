cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)
# Single source contains host and device code: compile it with nvcc
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# fmad=false keeps arithmetic identical to the unfused reference expression
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -O3 -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX CUDA::cudart m)
target_link_libraries(unstructured PRIVATE gomp)
