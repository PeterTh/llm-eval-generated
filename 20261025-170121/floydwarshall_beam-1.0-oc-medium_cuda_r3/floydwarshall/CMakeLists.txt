cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

target_link_libraries(floydwarshall m cudart)
