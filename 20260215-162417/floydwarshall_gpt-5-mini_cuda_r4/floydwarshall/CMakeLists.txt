cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Treat .cpp as CUDA source so nvcc compiles it
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for C++ and CUDA
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# Link math library
target_link_libraries(floydwarshall m)
