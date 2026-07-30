cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

add_executable(stencil3d stencil3d.cpp)

# Compile the source file with CUDA compiler
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Target GPU architecture (sm_86 = RTX 3090 / Ampere; generates fatbin for sm_86 + PTX)
set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations (nvcc passes -O3 through to both host and device compiler)
target_compile_options(stencil3d PRIVATE -O3 --use_fast_math)

# Link math library
target_link_libraries(stencil3d m)
