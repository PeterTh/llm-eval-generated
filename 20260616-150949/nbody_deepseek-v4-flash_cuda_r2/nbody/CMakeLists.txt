cmake_minimum_required(VERSION 3.12)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cu)

# Enable optimizations for host compiler
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Set CUDA architecture for broad GPU compatibility (Pascal and later)
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
)

# CUDA compiler flags
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
