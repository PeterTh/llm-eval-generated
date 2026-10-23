cmake_minimum_required(VERSION 3.18)

# Build native cubins for common GPU architectures; PTX is embedded as well so
# the kernel also runs on newer/unknown devices.  This must be set before the
# CUDA language is enabled below.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the single source as CUDA.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations for both device and host code.
target_compile_options(black_scholes PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(black_scholes m)
