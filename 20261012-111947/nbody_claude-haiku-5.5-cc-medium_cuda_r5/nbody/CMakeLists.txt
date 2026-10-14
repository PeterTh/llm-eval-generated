cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on this machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(nbody nbody.cpp)

# nbody.cpp contains device kernels, so it is compiled by nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host flags are forwarded to the host compiler via -Xcompiler)
target_compile_options(nbody PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(nbody m)
