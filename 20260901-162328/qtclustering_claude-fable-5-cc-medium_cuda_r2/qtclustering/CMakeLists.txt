cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(qtclustering qtclustering.cpp)

# The source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# -fmad=false keeps device distance computations bit-identical to the
# plainly evaluated host expression, preserving the sequential results.
target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false --expt-relaxed-constexpr -Xcompiler=-O3,-Wall,-Wextra>)

# Link math library
target_link_libraries(qtclustering m)
