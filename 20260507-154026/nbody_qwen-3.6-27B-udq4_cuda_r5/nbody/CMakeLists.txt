cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra -lineinfo)

# Link math library
target_link_libraries(nbody m)
