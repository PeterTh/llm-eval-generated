cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(roomsim roomsim.cu)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)

# Link math library
target_link_libraries(roomsim m)
