cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile for the GPU(s) present on the build machine.  A caller-provided
# value (e.g. -DCMAKE_CUDA_ARCHITECTURES=86) still takes precedence.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(unstructured unstructured.cpp)

# The benchmark is implemented with the CUDA runtime, so compile the existing
# source file with nvcc (keeping the executable and source names unchanged).
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# Link math library
target_link_libraries(unstructured m)
