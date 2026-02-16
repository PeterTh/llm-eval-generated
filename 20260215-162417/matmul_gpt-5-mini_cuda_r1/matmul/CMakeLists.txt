cmake_minimum_required(VERSION 3.10)
# Set target CUDA architectures to avoid auto-detection failures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Treat matmul.cpp as a CUDA source so NVCC compiles it
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations for host and device
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library (CUDA runtime is linked automatically)
target_link_libraries(matmul PRIVATE m)
