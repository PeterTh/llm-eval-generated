cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(black_scholes black_scholes.cpp)

# Compile the source as CUDA (kernels live in the .cpp file)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extra-device-vectorization -Xcompiler=-O3,-march=native,-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(black_scholes m)
