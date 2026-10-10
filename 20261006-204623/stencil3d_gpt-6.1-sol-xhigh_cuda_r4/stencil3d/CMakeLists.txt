cmake_minimum_required(VERSION 3.24)

# Target the installed GPU by default; callers can specify architectures for
# cross-compilation with -DCMAKE_CUDA_ARCHITECTURES=... .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing filename and executable name while compiling with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# Preserve double-precision arithmetic without fast-math reassociation.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(stencil3d m)
