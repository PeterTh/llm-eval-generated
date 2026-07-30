cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

# Compile the .cpp with nvcc so CUDA kernels are recognised
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Target the RTX 3090 compute capability (SM 8.6)
set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES "86")

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(qtclustering m)
