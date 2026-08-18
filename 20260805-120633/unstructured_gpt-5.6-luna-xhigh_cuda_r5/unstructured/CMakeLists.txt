cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark is intentionally a CUDA-only executable.  Keep the existing
# source name, but compile it as CUDA so the kernel and host driver remain in
# one translation unit.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# The supplied benchmark machine is an Ampere GPU.  A native, optimized code
# path avoids carrying unrelated architectures into this performance target.
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(unstructured unstructured.cpp)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations for both the CUDA device code and the host driver.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

target_link_libraries(unstructured PRIVATE CUDA::cudart m)
