cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture (fallback to a common one if not detected)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES 60)

# Enable optimizations
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_60> -O3 -Wall -Wextra)

# Link math library
target_link_libraries(unstructured m)
