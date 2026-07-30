cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for performance
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -lineinfo)

# Link math library
target_link_libraries(spmv m)
