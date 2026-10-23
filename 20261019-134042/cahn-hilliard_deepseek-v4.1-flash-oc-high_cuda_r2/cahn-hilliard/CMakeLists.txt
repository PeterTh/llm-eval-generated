cmake_minimum_required(VERSION 3.18)

# Build for the common NVIDIA compute capabilities (with PTX for forward
# compatibility) so the binary runs on a wide range of GPUs.
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90" CACHE STRING "CUDA architectures")

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# The benchmark source is compiled as CUDA (it is still named .cpp).
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations for the host compiler
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Enable optimizations for the device compiler
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extra-device-vectorization>)

# Link CUDA runtime and math library
target_link_libraries(cahn_hilliard PRIVATE CUDA::cudart m)
