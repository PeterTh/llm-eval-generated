cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cu)

# Enable optimizations for both host and device
set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "native"
)
target_compile_options(black_scholes PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link math library
target_link_libraries(black_scholes m)
