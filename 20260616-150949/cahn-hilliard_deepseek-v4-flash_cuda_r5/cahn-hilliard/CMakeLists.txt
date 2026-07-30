cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile .cpp as CUDA source
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Target GPU architecture (RTX 3090 = sm_86)
set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -Wall -Wextra)

# CUDA-specific flags
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Link math library
target_link_libraries(cahn_hilliard m)
