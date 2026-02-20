cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>)

# Link libraries
target_link_libraries(roomsim m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
