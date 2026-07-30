cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(nbody nbody.cpp)

# Treat .cpp as CUDA source
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for CUDA
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(nbody m)
