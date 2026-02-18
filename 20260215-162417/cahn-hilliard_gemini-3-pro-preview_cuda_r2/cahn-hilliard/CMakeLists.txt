cmake_minimum_required(VERSION 3.10)

find_program(CUDA_COMPILER_PATH nvcc HINTS /usr/local/cuda/bin /usr/local/cuda-12.6/bin)
if(CUDA_COMPILER_PATH)
    set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_PATH})
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>)
else()
    target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link math library
target_link_libraries(cahn_hilliard m)
