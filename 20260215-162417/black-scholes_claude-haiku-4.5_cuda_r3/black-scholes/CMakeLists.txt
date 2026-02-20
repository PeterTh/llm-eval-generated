cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Set source file as CUDA
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall>)

# Link math library and CUDA libraries
target_link_libraries(black_scholes m)
