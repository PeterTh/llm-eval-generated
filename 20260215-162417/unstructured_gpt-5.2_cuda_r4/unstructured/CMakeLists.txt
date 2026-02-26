cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Compile this existing source with NVCC so we can use CUDA kernels without adding new files.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Prefer building for the local GPU for maximum performance.
set_property(TARGET unstructured PROPERTY CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

# Link math + CUDA runtime
target_link_libraries(unstructured PRIVATE m CUDA::cudart)
