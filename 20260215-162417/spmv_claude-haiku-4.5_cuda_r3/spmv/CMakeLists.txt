cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture - supports compute capability 6.0 and newer
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

# Set CUDA optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_executable(spmv spmv.cu)

# Enable optimizations for C++
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(spmv m)
