cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Target multiple GPU architectures (Kepler through Hopper)
set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES "52;61;70;75;80;86;90")

target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
target_link_libraries(matmul m)
