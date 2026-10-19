cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 $<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

# Link math library
target_link_libraries(matmul m)
