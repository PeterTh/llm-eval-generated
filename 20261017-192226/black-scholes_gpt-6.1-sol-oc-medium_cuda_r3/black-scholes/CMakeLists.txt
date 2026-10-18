cmake_minimum_required(VERSION 3.24)

# Target the build machine's GPU unless the caller specifies architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Preserve double-precision arithmetic and expression rounding; fast math is
# deliberately avoided for the normal CDF and cancellation-sensitive prices.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(black_scholes m)
