cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "80;86")
target_compile_options(stencil3d PRIVATE -O3 -Wall -Wextra -lineinfo)

# Link math and CUDA runtime libraries
target_link_libraries(stencil3d m cudart pthread)
