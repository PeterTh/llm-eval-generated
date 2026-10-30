cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(stencil3d m)