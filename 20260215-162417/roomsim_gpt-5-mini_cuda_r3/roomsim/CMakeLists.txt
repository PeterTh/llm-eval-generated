cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

add_executable(roomsim roomsim.cpp)

# Treat the .cpp as CUDA source so NVCC compiles it
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
# CUDA-specific compile options
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# Link math and CUDA runtime
target_link_libraries(roomsim m cuda)
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
