cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(unstructured unstructured.cu)
set_target_properties(unstructured PROPERTIES LINKER_LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "86")

# Link math library
target_link_libraries(unstructured m cuda)
