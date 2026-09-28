cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark CXX CUDA)

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

# The benchmark source is compiled as CUDA (kernels live in the same file).
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE
  # --fmad=false keeps the flux accumulation bit-identical to the scalar reference
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(unstructured m)
