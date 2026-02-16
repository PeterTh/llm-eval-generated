cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_FLAGS "-O3")
target_link_libraries(nbody PRIVATE cudart m)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE CUDA::cuda CUDA::cudart m)
