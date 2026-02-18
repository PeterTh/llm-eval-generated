cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
endif()

# CUDA specific flags
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(floydwarshall m)
