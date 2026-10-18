cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source and executable names, compiling the implementation
# with nvcc. CUDA is required; there is no serial execution path.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(stencil3d m)
