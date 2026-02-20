cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
enable_language(CUDA)

# Rename source to .cu for CUDA compilation
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# Set CUDA architectures (adjust based on target GPU)
set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")

target_compile_options(qtclustering PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link math library
target_link_libraries(qtclustering m)
