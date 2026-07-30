cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (use computed capability from device, or default to a broad range)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

add_executable(roomsim roomsim.cpp)

# Treat the source file as CUDA so nvcc compiles it
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# CUDA compiler flags: fast math, optimize
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic")

# Link math library
target_link_libraries(roomsim m)
