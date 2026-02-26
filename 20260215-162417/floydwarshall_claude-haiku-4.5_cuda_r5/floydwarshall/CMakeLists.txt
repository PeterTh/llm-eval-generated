cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)

# Set file to be compiled as CUDA
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations for both C++ and CUDA
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-march=native>)

# Set CUDA architecture
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "70;75;80;90")

# Link math library and CUDA
target_link_libraries(floydwarshall m)
