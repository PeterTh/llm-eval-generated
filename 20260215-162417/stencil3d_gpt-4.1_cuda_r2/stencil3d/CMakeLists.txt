cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)


# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -Wall -Wextra)

# Include CUDA headers
include_directories(/usr/local/cuda/include)

# Link CUDA runtime and math library
target_link_libraries(stencil3d cudart m)

set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
