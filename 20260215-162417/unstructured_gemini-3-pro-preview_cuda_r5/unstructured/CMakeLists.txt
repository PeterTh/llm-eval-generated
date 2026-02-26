cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
if(NOT MSVC)
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)
endif()

set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Link math library
target_link_libraries(unstructured m)
