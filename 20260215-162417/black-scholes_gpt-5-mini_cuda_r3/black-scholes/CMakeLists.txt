cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language and set architecture for broad compatibility
enable_language(CUDA)

add_executable(black_scholes black_scholes.cpp)

# Tell CMake to compile the .cpp file with the CUDA compiler
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable separable compilation for better optimization
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compiler flags for CPU and CUDA
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--prec-div;--prec-sqrt>
)

# Set CUDA architectures if supported (leave CUDA to decide if not)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 90)
endif()

# Link math and CUDA runtime libraries
target_link_libraries(black_scholes m cuda cudart)

