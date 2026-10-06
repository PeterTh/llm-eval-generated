cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The source is compiled as CUDA
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Enable optimizations (no FMA contraction on device, to match host results)
target_compile_options(black_scholes PRIVATE
    -O3 --fmad=false -Xcompiler=-O3,-march=native,-Wall,-Wextra)

# Link math library and CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(black_scholes CUDA::cudart m)
