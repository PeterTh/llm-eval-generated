cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark contains CUDA kernels; compile it with nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# The kernel contracts its multiply-add into an FMA, which is more accurate than
# the sequential version but not bit-identical to it for every matrix size.
# Turning this on reproduces the sequential results bit for bit, at ~2x the cost.
option(MATMUL_NO_FMA "Disable fused multiply-add in the CUDA kernel" OFF)
if(MATMUL_NO_FMA)
    target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
endif()

# Link math library
target_link_libraries(matmul m)
