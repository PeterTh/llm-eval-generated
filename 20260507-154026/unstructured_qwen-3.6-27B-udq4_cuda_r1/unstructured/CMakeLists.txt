cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set(CUDA_ARCHITECTURES 86)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_options(unstructured PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)

# Link math library
target_link_libraries(unstructured m)
