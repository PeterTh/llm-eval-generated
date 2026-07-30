cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

enable_language(CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cu)

# Set CUDA standard and architecture
set_target_properties(black_scholes PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES OFF
)

# Enable optimizations for both host and device code
target_compile_options(black_scholes PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link math library
target_link_libraries(black_scholes m)

# Include common headers
target_include_directories(black_scholes PRIVATE ../common)
