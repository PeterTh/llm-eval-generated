cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Auto-detect GPU architecture
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES native)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3)

# Link math library
target_link_libraries(floydwarshall m)
