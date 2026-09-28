cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

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

# The single source file is CUDA code (contains kernels); compile it with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations; forward OpenMP to the host compiler behind nvcc.
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP runtime, CUDA runtime, and math library
target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
