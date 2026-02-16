cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile roomsim.cpp with CUDA (nvcc) to enable device kernels in the same file
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings for host and device
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math and CUDA runtime libraries
target_link_libraries(roomsim PRIVATE m -lcudart)
