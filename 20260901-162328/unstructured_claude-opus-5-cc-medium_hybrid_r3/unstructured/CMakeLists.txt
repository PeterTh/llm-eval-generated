cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The single translation unit contains host and device code
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET unstructured PROPERTY CUDA_ARCHITECTURES 70 80 86 90)
endif()

# Enable optimizations. --fmad=false keeps the device arithmetic bit-identical
# to the serial reference implementation.
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false --expt-relaxed-constexpr
     -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
