cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename source to .cu for nvcc compilation
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
set_target_properties(roomsim PROPERTIES
    CUDA_ARCHITECTURES "80"
)
target_compile_options(roomsim PRIVATE -O3 -Wall -Wextra)

# Link math and CUDA runtime libraries
target_link_libraries(roomsim m)
