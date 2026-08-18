cmake_minimum_required(VERSION 3.24)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile for the RTX 3090/compute capability 8.6 used by this benchmark host.
# The cache variable allows a caller to select a different CUDA target.
set(STENCIL3D_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures for stencil3d")

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES "${STENCIL3D_CUDA_ARCHITECTURES}")

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(stencil3d PRIVATE m)
