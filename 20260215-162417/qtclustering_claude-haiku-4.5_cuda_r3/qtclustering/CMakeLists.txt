cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture targets
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Treat .cpp file as CUDA source
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link CUDA runtime and math library
target_link_libraries(qtclustering PRIVATE m)
