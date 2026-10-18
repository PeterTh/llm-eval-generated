cmake_minimum_required(VERSION 3.18)
# Target the local GPU by default when supported, while allowing callers to
# select architectures explicitly for cross-compilation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(nbody m)
