cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cu)

# Set CUDA architecture for RTX 3090 (Ampere, sm_86)
set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(cahn_hilliard m)
