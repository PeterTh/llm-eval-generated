cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
add_executable(stencil3d stencil3d.cu)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -arch=sm_60 -Xcompiler "-Wall -Wextra")

# Link math library
target_link_libraries(stencil3d m cuda cudart)
