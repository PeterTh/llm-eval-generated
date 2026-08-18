cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark host uses Ampere GPUs (compute capability 8.6).  Emit native
# SASS for that target so the timed path never pays for PTX JIT compilation.
set(CMAKE_CUDA_ARCHITECTURES 86)

# The source keeps its original name, but is compiled by nvcc because it
# contains the CUDA kernel and launch configuration.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable host and device optimizations.  CUDA is intentionally required: the
# executable has no CPU fallback implementation.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
