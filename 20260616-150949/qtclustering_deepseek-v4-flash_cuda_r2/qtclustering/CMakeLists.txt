cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target RTX 3090 compute capability (sm_86)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(qtclustering qtclustering.cpp)

# Compile qtclustering.cpp as CUDA source so NVCC handles the kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and CUDA runtime
target_link_libraries(qtclustering m cuda)
