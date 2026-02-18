cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture if not already specified via command line
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Set CUDA compilation flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Set CUDA compilation for the source file
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Link math library and CUDA runtime
target_link_libraries(matmul m cuda)
