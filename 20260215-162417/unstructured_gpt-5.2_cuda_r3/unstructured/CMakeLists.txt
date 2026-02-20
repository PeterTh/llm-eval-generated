cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Max performance on the build machine's GPU.
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(unstructured unstructured.cpp)

# Compile the existing .cpp as CUDA (no new files).
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(unstructured PRIVATE CUDA::cudart m)
