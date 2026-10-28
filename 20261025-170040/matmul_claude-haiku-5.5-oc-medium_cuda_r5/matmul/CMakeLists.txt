cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# matmul.cpp is compiled as CUDA source so it can contain device kernels.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations (host code is forwarded to the host compiler via nvcc).
# --fmad=false keeps every multiply and add separately rounded, matching the
# source semantics of `sum += a * b` so GPU results are bit-identical to a
# non-contracted CPU reference. Remove it for FMA throughput at the cost of
# last-bit differences.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(matmul m)
