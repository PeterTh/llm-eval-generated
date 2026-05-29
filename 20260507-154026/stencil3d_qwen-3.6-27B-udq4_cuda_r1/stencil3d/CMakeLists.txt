cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(stencil3d stencil3d.cpp)

set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(stencil3d PRIVATE -O3 -Wall -Wextra)
target_link_libraries(stencil3d m)
