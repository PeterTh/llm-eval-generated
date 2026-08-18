cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep the established source layout and executable name, but compile the
# implementation with nvcc so its CUDA kernel is always used.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
# Let nvcc select its default device code target.  This keeps the build usable
# across the different GPUs used for this benchmark without a runtime CPU path.
set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES OFF)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(stencil3d m)
