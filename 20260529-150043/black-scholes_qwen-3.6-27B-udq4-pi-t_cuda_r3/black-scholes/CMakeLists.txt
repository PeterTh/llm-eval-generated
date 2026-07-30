cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for maximum performance
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect if not set; fallback to common architectures
    set(CMAKE_CUDA_ARCHITECTURES 50 60 61 70 75 80 86 89 90)
endif()

add_executable(black_scholes black_scholes.cpp)

# Force compilation with nvcc since the file contains CUDA device code
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both host and device code
set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(black_scholes PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link math library
target_link_libraries(black_scholes m)
