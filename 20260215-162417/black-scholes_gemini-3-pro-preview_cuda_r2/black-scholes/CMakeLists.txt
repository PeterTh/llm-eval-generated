cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin)
    if(CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER})
    endif()
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(black_scholes m)
