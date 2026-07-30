cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Force CUDA compilation for cholesky.cpp
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(cholesky PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -Wall -Wextra)

# CUDA-specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Link math library
target_link_libraries(cholesky PRIVATE m)
