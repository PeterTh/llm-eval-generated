cmake_minimum_required(VERSION 3.17)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark is run on the CUDA device, so CUDA is a required build
# language.  86 is the native architecture of the benchmark GPU; callers can
# still override this cache variable for another CUDA-capable device.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for")
endif()

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both the CUDA device and the host portions of the
# single CUDA translation unit.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-march=native>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(black_scholes PRIVATE CUDA::cudart m)
