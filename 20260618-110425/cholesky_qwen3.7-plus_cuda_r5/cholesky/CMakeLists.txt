cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cholesky cholesky.cpp)

# Treat .cpp file as CUDA code
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -arch=sm_86>)

# Link CUDA runtime and math library
target_link_libraries(cholesky cudart m)
