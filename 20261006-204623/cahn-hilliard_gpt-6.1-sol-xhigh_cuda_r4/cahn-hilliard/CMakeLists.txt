cmake_minimum_required(VERSION 3.18)

# Prefer code for the installed GPU, while allowing CUDAARCHS or an explicit
# CMAKE_CUDA_ARCHITECTURES to select targets for builds on another machine.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24
   AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling it as CUDA.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Retain the original floating-point evaluation order (no fused operations).
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>
)

# Link math library
target_link_libraries(cahn_hilliard m)
