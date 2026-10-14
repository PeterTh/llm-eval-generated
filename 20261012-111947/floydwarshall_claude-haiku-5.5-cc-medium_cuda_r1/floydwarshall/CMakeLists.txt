cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures; override with -DCMAKE_CUDA_ARCHITECTURES=<list>
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

add_executable(floydwarshall floydwarshall.cpp)

# The source is compiled by nvcc (it contains the CUDA kernel)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host flags are forwarded to the C++ compiler through nvcc)
target_compile_options(floydwarshall PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link math library
target_link_libraries(floydwarshall m)
