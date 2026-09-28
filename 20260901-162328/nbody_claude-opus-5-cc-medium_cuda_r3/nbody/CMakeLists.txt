cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(nbody nbody.cpp)

# The source is CUDA C++ and must be compiled by nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations. Fused multiply-add is allowed (as it is for the host
# compiler with -O3 -march=native), fast-math is not: all device arithmetic
# stays IEEE double precision.
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=true -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(nbody m)
