cmake_minimum_required(VERSION 3.24)

# Select the installed GPU by default; callers can still override this cache
# variable for cross-compilation.  This must precede project() so CMake does not
# cache nvcc's conservative virtual-architecture default first.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the requested existing filename while compiling it as CUDA source.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Strict double-precision operation ordering keeps CPU-equivalent numerical
# semantics; optimization and native code generation remain fully enabled.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--extra-device-vectorization;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(unstructured m)
