cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) of this machine unless architectures are given explicitly
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    endif()
endif()

add_executable(roomsim roomsim.cpp)

# The single source file is compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings. The host code keeps the floating-point
# contraction behaviour of the original GNU C++ build (mesh generation must
# produce bit-identical geometry).
target_compile_options(roomsim PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra
)

# Link math library
target_link_libraries(roomsim m)
