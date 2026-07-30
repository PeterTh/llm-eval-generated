cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Compile with CUDA
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -Wall -Wextra)
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Target RTX 3090 (sm_86)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "86")

# Link math library
target_link_libraries(unstructured m)
