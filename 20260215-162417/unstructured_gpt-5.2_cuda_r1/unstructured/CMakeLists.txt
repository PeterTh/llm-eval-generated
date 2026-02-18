cmake_minimum_required(VERSION 3.10)

# Prefer nvcc when available; otherwise use clang++ for CUDA.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(NVCC nvcc)
  if(NVCC)
    set(CMAKE_CUDA_COMPILER "${NVCC}" CACHE FILEPATH "" FORCE)
  else()
    find_program(CLANGPP clang++ REQUIRED)
    set(CMAKE_CUDA_COMPILER "${CLANGPP}" CACHE FILEPATH "" FORCE)
    set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "" FORCE)
  endif()
endif()

# Avoid CMake failing to auto-detect a default architecture in headless environments.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "" FORCE)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set(UNSTRUCTURED_CUDA_FLAGS -O3 --std=c++20)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  list(APPEND UNSTRUCTURED_CUDA_FLAGS -march=native)
else()
  list(APPEND UNSTRUCTURED_CUDA_FLAGS -Xcompiler=-march=native)
endif()

target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:${UNSTRUCTURED_CUDA_FLAGS}>
)

# Link math library
target_link_libraries(unstructured m)
