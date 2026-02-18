cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set .cpp files to be compiled as CUDA
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Add CUDA include directories
target_include_directories(unstructured PRIVATE /usr/local/cuda-12.6/include)

# Enable optimizations
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

# CUDA-specific optimization flags
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler> $<$<COMPILE_LANGUAGE:CUDA>:-Wall> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler> $<$<COMPILE_LANGUAGE:CUDA>:-Wextra>)

# Link math library and CUDA
target_link_libraries(unstructured m)
target_link_libraries(unstructured /usr/local/cuda-12.6/lib64/libcudart.so)
