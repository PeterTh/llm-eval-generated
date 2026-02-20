cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile .cpp file as CUDA
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)

# Use native GPU architecture if CMake supports it, otherwise common architectures
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "native")
else()
    set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")
endif()

target_link_libraries(stencil3d m)
