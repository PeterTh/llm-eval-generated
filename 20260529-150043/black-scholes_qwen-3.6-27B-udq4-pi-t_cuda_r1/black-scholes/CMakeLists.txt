cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the .cpp source with nvcc so that CUDA device code is included
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Host compiler flags
set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES OFF
)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA-specific flags for maximum performance
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;-lineinfo>
)

# Link math library
target_link_libraries(black_scholes m)
