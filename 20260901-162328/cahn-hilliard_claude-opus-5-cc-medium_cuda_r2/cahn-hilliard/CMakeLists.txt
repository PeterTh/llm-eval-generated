cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) present in the machine; fall back to a broad set of
# architectures on CMake versions that do not support "native".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
endif()

# The benchmark source is CUDA code; keep the original file name.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(cahn_hilliard m)
