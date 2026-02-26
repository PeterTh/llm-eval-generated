cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA for this project
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)

# Compile options
set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "-O3 --expt-extended-lambda --expt-relaxed-constexpr")

add_executable(qtclustering qtclustering.cu)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LANGUAGE CUDA
)

# Link math library
target_link_libraries(qtclustering m)
