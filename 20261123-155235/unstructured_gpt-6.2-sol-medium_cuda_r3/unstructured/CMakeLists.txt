cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CUDA)

# Compile the existing source as CUDA while retaining the executable name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Keep the reference's rounding outside the explicit fused operation in the kernel.
target_compile_options(unstructured PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)
