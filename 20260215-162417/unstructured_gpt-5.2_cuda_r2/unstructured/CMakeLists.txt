cmake_minimum_required(VERSION 3.10)

# Use clang as the CUDA compiler when nvcc isn't available (common in CI images).
if (NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER clang++)
endif()

# CMake can't always auto-detect an architecture in headless CI; build a portable fatbin.
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Compile this TU with NVCC so the benchmark unconditionally uses CUDA.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  -Wall -Wextra)

find_package(CUDAToolkit REQUIRED)

# Link math + CUDA runtime
target_link_libraries(unstructured PRIVATE m CUDA::cudart)
