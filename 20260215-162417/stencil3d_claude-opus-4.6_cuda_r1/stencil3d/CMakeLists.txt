cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Treat .cpp as CUDA source
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(stencil3d m)
