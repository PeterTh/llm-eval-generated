cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Compile as CUDA source
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -Wall -Wextra)

# Set CUDA architecture for RTX 3090
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "86")

# Link math library
target_link_libraries(nbody m)
