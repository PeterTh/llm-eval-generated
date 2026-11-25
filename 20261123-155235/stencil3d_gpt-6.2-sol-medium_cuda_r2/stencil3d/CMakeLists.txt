cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
project(stencil3d_benchmark LANGUAGES CUDA)

set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(stencil3d PRIVATE -O3 -Xcompiler=-O3,-Wall,-Wextra)
