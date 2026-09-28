cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPUs present on this machine; override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(roomsim roomsim.cpp)

# The source keeps its .cpp name but is CUDA code
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings (host-side flags passed through nvcc)
target_compile_options(roomsim PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(roomsim m)
