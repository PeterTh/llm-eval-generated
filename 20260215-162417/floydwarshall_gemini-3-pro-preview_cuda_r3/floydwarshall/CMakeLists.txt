cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile .cpp file as CUDA
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -O3 -arch=native>)
else()
    target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link math library
target_link_libraries(floydwarshall m)
