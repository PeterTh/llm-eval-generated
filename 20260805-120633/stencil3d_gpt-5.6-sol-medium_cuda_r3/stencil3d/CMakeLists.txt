cmake_minimum_required(VERSION 3.24)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling it as CUDA.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# Build native SASS for the installed GPU. The benchmark is intentionally an
# unconditional CUDA implementation and therefore requires a CUDA-capable GPU.
set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES native)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xptxas=-O3;-Xcompiler=-Wall,-Wextra>
)

target_link_libraries(stencil3d PRIVATE m)
