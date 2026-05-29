cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(qtclustering qtclustering.cpp)

# Compile .cpp file with nvcc since it contains CUDA code
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(qtclustering m)
