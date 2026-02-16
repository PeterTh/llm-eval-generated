cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Enable optimizations for C++ and CUDA
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Compile the .cpp with NVCC so CUDA kernels can live in the same file
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)
# Link CUDA runtime and math library
target_link_libraries(nbody PRIVATE CUDAToolkit::cudart m)
