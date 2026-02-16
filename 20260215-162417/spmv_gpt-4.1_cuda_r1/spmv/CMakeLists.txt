cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)

# Link math and CUDA libraries
target_link_libraries(spmv PRIVATE m cuda cudart)
