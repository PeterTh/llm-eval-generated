cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)
# Force CMake to treat the .cpp as CUDA source so NVCC compiles it
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both host and device
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)
# CUDA-specific flags
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

# Link math library
target_link_libraries(black_scholes m)

# Enable separable compilation for CUDA
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
