cmake_minimum_required(VERSION 3.24)
# Optimize for the installed GPU unless the caller specifies target architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

# Compile the existing source as CUDA without renaming it or the executable.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# Keep IEEE double arithmetic and the original expression evaluation order.
# CMAKE_CUDA_ARCHITECTURES can be set by the caller for the target GPU.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
