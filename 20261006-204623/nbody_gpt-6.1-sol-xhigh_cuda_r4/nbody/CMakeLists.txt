cmake_minimum_required(VERSION 3.24)

# Optimize for the installed GPU unless a cross-compilation target is supplied.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename, but compile the simulation with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(nbody m)
