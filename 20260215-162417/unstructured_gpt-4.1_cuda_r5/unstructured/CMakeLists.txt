cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -Wall -Wextra)

# Link math library and CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(unstructured m CUDA::cudart)
