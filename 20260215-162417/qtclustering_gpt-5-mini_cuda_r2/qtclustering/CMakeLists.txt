cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

# Set a default CUDA architecture to avoid auto-detection failures
set(CMAKE_CUDA_ARCHITECTURES 60)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure the .cpp is compiled with the CUDA compiler so device kernels can live in the same file
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# Enable separable compilation for CUDA
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(qtclustering m)
