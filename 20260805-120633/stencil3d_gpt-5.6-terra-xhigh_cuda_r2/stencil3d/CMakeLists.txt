cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA so the benchmark always uses the GPU path.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# Build specifically for the GPU present on the benchmark host; this avoids
# runtime JIT compilation and permits native Ampere code generation.
set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES 86)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(stencil3d PRIVATE CUDA::cudart)
