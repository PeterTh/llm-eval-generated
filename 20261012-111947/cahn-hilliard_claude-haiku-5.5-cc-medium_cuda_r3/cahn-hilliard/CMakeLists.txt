cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build device code for the GPU architecture(s) present on this machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The benchmark is GPU-only: compile the source as CUDA
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host code is forwarded to the host compiler through -Xcompiler).
# --fmad=false keeps every multiply and add separately rounded, matching the CPU reference bit-for-bit.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(cahn_hilliard m)
