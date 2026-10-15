cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)
# The source is compiled as CUDA (contains device code)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp,-Wall,-Wextra>)
# Keep IEEE semantics of the reference (no fast-math); override the flag above
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false --prec-div=true --prec-sqrt=true --fmad=true>)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
