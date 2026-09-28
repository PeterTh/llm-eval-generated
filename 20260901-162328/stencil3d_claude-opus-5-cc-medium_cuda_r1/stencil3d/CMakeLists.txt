cmake_minimum_required(VERSION 3.24)

# Build for the GPU(s) of the build machine unless told otherwise. This has to
# happen before project(), which would otherwise pick the (very old) default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark source contains CUDA kernels; compile it with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations. Note that fast-math must stay off: the stencil has to
# reproduce the exact IEEE-754 double results of the reference implementation.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(stencil3d m)
