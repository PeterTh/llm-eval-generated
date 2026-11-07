cmake_minimum_required(VERSION 3.18)
# CUDA architectures must be chosen before the CUDA language is enabled
# Target the GPUs present on the build machine unless the user specifies otherwise
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  endif()
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The single source file contains host code, OpenMP regions and CUDA kernels
add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(unstructured PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Enable optimizations. -fmad=false keeps the device arithmetic identical to
# the original host expression order (bit-reproducible results).
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false --expt-relaxed-constexpr
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(unstructured PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} OMPI_SKIP_MPICXX)

# Link MPI, OpenMP, the CUDA runtime and the math library
target_link_libraries(unstructured PRIVATE
  ${MPI_CXX_LIBRARIES}
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m)
target_link_options(unstructured PRIVATE -fopenmp)
