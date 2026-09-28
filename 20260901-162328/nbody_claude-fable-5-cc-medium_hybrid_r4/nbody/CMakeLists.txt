cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The single source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations; forward host-compiler flags (incl. OpenMP) through nvcc
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI/OpenMP runtime libraries directly (nvcc drives the link step)
target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
