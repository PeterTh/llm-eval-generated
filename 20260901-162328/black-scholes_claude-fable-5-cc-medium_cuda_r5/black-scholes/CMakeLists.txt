cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# The source is CUDA code kept in a .cpp file
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Build for the GPUs present on this machine
set_target_properties(black_scholes PROPERTIES CUDA_ARCHITECTURES native)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(black_scholes m)
