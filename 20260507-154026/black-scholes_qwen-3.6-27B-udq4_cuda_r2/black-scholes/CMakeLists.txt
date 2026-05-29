cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Work around GCC 13 / CUDA 12.0 _Float32 incompatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -allow-unsupported-compiler -Xcompiler -fPIC,-D_FORCE_INLINES")

add_executable(black_scholes black_scholes.cu)
target_link_libraries(black_scholes PRIVATE m)
