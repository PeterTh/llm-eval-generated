cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

target_link_libraries(matmul m cudart)
