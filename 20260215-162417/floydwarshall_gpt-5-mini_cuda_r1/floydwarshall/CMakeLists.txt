cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow compiling .cpp files with the CUDA compiler so the single source can contain CUDA kernels
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS "cu;cpp;cc;cxx")

add_executable(floydwarshall floydwarshall.cpp)

find_package(CUDAToolkit REQUIRED)

# Optimizations for both host and CUDA device
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)

target_link_libraries(floydwarshall PRIVATE m CUDA::cudart)
