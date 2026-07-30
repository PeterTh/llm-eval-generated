cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use compute capability 8.0 (Turing) as a good baseline;
# the binary will also run on newer architectures.
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(stencil3d stencil3d.cpp)

# Force nvcc to compile the .cpp file (it contains CUDA kernels)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for device code
target_compile_options(stencil3d PRIVATE -O3 -Xptxas -O3 -lineinfo)

# Link math library
target_link_libraries(stencil3d m)
