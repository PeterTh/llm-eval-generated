cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set(CMAKE_CUDA_ARCHITECTURES 86)
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Link math library and CUDA runtime
target_link_libraries(unstructured m cuda cudart)
