cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -Wall -Wextra)
set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")

# Link math library
target_link_libraries(stencil3d m)
