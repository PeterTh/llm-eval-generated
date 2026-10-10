cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without changing the executable name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Preserve the original double-precision operation order (no contraction or
# fast-math). Allow callers to select architectures with CMAKE_CUDA_ARCHITECTURES.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
target_link_libraries(unstructured PRIVATE m)
