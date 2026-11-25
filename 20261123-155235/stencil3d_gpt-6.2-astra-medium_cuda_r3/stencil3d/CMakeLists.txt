cmake_minimum_required(VERSION 3.20)
project(stencil3d_benchmark LANGUAGES CUDA)

set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(stencil3d PRIVATE -O3)
