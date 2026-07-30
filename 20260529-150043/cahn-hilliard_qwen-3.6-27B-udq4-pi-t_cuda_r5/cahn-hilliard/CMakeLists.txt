cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architectures
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "50;60;61;70;75;80;86")

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(cahn_hilliard m)
