cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler optimizations
set_target_properties(black_scholes PROPERTIES
    CXX_COMPILE_OPTIONS "-O3 -march=native -Wall -Wextra"
)

# CUDA compiler optimizations
set_target_properties(black_scholes PROPERTIES
    CUDA_COMPILE_OPTIONS "-O3 -Xcompiler=-O3,-march=native"
)

# Link math library
target_link_libraries(black_scholes m)
