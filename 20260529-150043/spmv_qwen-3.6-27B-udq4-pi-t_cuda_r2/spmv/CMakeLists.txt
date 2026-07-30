cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)
target_compile_options(spmv PRIVATE --expt-relaxed-constexpr)

# Link math library
target_link_libraries(spmv m)
