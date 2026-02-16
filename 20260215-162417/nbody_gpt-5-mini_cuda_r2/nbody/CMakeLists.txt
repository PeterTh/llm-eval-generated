cmake_minimum_required(VERSION 3.10)
# Preset architectures to assist environments where auto-detection may fail
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing .cpp as CUDA to enable device kernels in the same file
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations for host and device
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(nbody m)
