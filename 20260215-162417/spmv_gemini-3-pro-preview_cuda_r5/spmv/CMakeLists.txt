cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Set spmv.cpp to be compiled as CUDA code
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -O3 -arch=native>)
else()
    target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link math library
target_link_libraries(spmv m)
