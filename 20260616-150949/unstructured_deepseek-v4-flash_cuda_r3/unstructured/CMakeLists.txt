cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Compile as CUDA so nvcc processes device code and kernel launches
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(unstructured PROPERTIES
    CUDA_ARCHITECTURES "native"
)

# Enable optimizations (the source is compiled as CUDA, pass host flags via -Xcompiler)
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

# Link math library
target_link_libraries(unstructured m)
