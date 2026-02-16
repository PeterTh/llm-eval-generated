cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Treat floydwarshall.cpp as CUDA source so nvcc will compile it
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
# Enable CUDA-specific optimizations
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

# Link math library
target_link_libraries(floydwarshall m)
