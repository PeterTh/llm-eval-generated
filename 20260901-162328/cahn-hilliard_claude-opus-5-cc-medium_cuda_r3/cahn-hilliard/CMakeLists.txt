cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU in this machine unless told otherwise.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark source contains CUDA kernels; compile it with nvcc.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations. -fmad=false keeps the floating-point expressions unfused,
# so every result is bit-identical to the arithmetic written in the source.
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false
    -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(cahn_hilliard m)
