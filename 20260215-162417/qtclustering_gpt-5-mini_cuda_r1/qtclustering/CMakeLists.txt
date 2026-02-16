cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

# Compile options for host and device
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
)

# Ensure the .cpp is compiled with NVCC as CUDA source
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Find CUDA toolkit and link runtime
find_package(CUDAToolkit REQUIRED)

# Link math and CUDA runtime
target_link_libraries(qtclustering PRIVATE m CUDA::cudart)
