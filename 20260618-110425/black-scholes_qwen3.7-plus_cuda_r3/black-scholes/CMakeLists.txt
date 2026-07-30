cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the .cpp file as CUDA
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# CUDA optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3;
        -arch=sm_86;
        --use_fast_math;
        -Xcompiler;-O3;
        -Xcompiler;-march=native;
        -Xcompiler;-Wall;
        -Xcompiler;-Wextra
    >
)

# Link CUDA runtime and math library
target_link_libraries(black_scholes cudart m)
