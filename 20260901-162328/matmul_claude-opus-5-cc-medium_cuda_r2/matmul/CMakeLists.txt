cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark is implemented in CUDA; compile the translation unit with nvcc.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations.
# -fmad=false keeps the device multiply and add unfused, so the GPU reproduces the exact
# IEEE-754 result of the serial implementation (an FMA-contracted kernel is about 2x faster
# but can differ from it by one ulp per element).
target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(matmul m)
