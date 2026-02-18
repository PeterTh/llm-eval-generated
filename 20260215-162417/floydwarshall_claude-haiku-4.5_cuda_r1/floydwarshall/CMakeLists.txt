cmake_minimum_required(VERSION 3.17)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "75;80;86;90")
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Link math library and CUDA
target_link_libraries(floydwarshall m cuda)
