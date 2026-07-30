cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -Wall -Wextra)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "86")

# Link math library
target_link_libraries(floydwarshall m)
