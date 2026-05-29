cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Mark the .cpp file as CUDA source so nvcc compiles it
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations via nvcc flags
target_compile_options(floydwarshall PRIVATE -O3 -arch=sm_86 -Xcompiler=-Wall,-Wextra)

# Link math library
target_link_libraries(floydwarshall m)
