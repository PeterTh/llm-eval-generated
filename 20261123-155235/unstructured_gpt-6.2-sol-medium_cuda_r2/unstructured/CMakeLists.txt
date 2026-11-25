cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CUDA)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Keep the flux multiplications and additions rounded separately.
target_compile_options(unstructured PRIVATE -O3 --fmad=false -Xcompiler=-O3,-Wall,-Wextra)
