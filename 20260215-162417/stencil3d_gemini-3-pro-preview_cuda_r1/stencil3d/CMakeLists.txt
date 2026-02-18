cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES native)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
if(MSVC)
  target_compile_options(stencil3d PRIVATE /O2)
else()
  target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Set CUDA architecture
set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES native)

# Link math library
target_link_libraries(stencil3d m)
