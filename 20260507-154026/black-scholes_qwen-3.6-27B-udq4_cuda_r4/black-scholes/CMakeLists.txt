cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cu)

set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations
set_target_properties(black_scholes PROPERTIES
    CUDA_compile_options "-O3 -Xcompiler=-march=native,-Wall,-Wextra,-funroll-loops"
)

# Link math library
target_link_libraries(black_scholes m)
