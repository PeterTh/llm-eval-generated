cmake_minimum_required(VERSION 3.18)

# Target the GPUs present at configure time when possible; otherwise build a
# portable fat binary covering common data-center / workstation architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
  endif()
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The single source file contains host (MPI/OpenMP) and device (CUDA) code and
# is compiled by the CUDA compiler.
add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable optimizations (host flags are forwarded through nvcc)
target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# MPI: use include dirs / libraries directly so that no host-only compile
# flags (e.g. -pthread) are passed to nvcc.
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(matmul PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
target_link_options(matmul PRIVATE ${MPI_CXX_LINK_FLAGS})
