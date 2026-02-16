cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

add_executable(unstructured unstructured.cpp)
# Tell CMake to compile this source with the CUDA compiler so kernels can be in .cpp file
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native>)

# Enable separable compilation for CUDA
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math and CUDA runtime
target_link_libraries(unstructured m CUDA::cudart)
