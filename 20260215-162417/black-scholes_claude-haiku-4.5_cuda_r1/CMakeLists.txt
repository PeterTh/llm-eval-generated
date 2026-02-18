cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure CUDA compute capability
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

add_executable(black_scholes black-scholes/black_scholes.cu)

# Enable optimizations
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(black_scholes m)

# Ensure CUDA libraries are linked
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

