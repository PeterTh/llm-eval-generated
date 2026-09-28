cmake_minimum_required(VERSION 3.18)

# CMAKE_CUDA_ARCHITECTURES must be set before CUDA is enabled (i.e. before project())
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the source as CUDA so the kernels defined in it can be built with nvcc
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(unstructured m)
