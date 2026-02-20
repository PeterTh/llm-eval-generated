cmake_minimum_required(VERSION 3.17)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA as the language for nbody.cpp
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations for CUDA (nbody.cpp is compiled as CUDA)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native> -Wall -Wextra)

# Set CUDA architecture (common targets: 52, 61, 70, 75, 80, 86, 89, 90)
# Using compute_52 for broader compatibility, but can be optimized for specific GPU
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "52;61;70;75;80;86")

# Link math library
target_link_libraries(nbody m)
