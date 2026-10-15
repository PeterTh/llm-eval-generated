cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The single source file contains host (MPI + OpenMP) and CUDA code
add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(cholesky PRIVATE ${MPI_CXX_LIBRARIES} CUDA::cudart -fopenmp m)
