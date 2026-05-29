cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(unstructured unstructured.cu)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(unstructured PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")

# Link math library
target_link_libraries(unstructured m)
