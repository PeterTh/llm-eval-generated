cmake_minimum_required(VERSION 3.24)

# Build for the GPU present on the benchmark host unless the caller explicitly
# requests a different set of CUDA architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source and executable names while compiling the source as
# CUDA so it can contain both the host driver and device kernel.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xptxas=-O3;-Xcompiler=-Wall,-Wextra>
)

target_link_libraries(black_scholes PRIVATE m)
