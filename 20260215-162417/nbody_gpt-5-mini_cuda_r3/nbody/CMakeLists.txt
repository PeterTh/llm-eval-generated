cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

# Provide a default CUDA architecture to avoid detection issues
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Treat the .cpp as CUDA source so nvcc compiles it
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations for host and device
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70 -lineinfo>
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-O3 -march=native -Wall -Wextra>
)

# Link math and CUDA runtime
target_link_libraries(nbody m cudart)
