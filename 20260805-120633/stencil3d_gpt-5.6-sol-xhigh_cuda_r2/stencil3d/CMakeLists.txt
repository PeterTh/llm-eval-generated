cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing filename while compiling it as CUDA source.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--restrict;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)

set_target_properties(stencil3d PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# CMake 3.24+ can ask nvcc to generate code for the installed GPU.  Older
# CMake versions retain nvcc's default architecture selection.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES native)
endif()

target_link_libraries(stencil3d PRIVATE m)
