cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep the existing source file and executable name, but compile the source as
# CUDA so the kernel and CUDA runtime calls are available.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
# Generate code for the GPU selected by the build host, avoiding a lower
# common architecture and runtime JIT overhead on the benchmark machine.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
    set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# Link math library
target_link_libraries(stencil3d m)
