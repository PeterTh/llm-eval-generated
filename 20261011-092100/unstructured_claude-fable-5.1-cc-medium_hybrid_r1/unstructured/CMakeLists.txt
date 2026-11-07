cmake_minimum_required(VERSION 3.18)

# Target the GPUs present on the build machine unless the user specified otherwise
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
  endif()
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The single source file contains host code, MPI calls, OpenMP regions and
# CUDA kernels: compile it with the CUDA compiler.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations (-fmad=false keeps the floating point expression
# evaluation identical to the original CPU code).
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false --expt-relaxed-constexpr -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Hybrid parallel runtime: MPI + OpenMP + CUDA runtime
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
