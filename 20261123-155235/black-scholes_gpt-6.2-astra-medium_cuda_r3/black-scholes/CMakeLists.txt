cmake_minimum_required(VERSION 3.20)

project(black_scholes_benchmark LANGUAGES CUDA)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(black_scholes PRIVATE -O3)
