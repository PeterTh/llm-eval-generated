cmake_minimum_required(VERSION 3.18)

# Prefer the toolkit under /usr/local/cuda (supports the default host compiler)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The whole program (host + device code) is compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings (host flags match the reference build,
# including GCC's default FMA contraction for GNU dialects)
target_compile_options(roomsim PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra
)

# Link math and thread libraries (one host thread drives each GPU)
find_package(Threads REQUIRED)
target_link_libraries(roomsim m Threads::Threads)
