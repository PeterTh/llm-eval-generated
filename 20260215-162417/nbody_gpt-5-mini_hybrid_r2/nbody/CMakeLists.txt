cmake_minimum_required(VERSION 3.10)

# Provide a default CUDA architecture before enabling CUDA language to avoid detection issues
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host and CUDA
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>)

# Link MPI, OpenMP and math libraries
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Enable CUDA separable compilation for better optimization
if (CMAKE_CUDA_COMPILER)
  set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
