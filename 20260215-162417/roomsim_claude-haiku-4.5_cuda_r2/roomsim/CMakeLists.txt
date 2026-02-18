cmake_minimum_required(VERSION 3.10)

# Set CUDA toolkit path before project declaration
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES LANGUAGE CUDA)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_60 -O3>)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wextra -Wpedantic>)

# Link math library and CUDA runtime
target_link_libraries(roomsim m cuda)

# Include directories
target_include_directories(roomsim PRIVATE /usr/local/cuda/include)
