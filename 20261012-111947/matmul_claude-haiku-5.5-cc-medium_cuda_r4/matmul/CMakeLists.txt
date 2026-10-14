cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build device code for the GPUs present on this machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(matmul matmul.cpp)

# matmul.cpp contains CUDA kernels, so compile it with nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host flags are forwarded to the host compiler via -Xcompiler).
# -fmad=false keeps the separate multiply and add rounding of the original `sum += a * b`,
# so device results match the CPU reference bit-for-bit.
target_compile_options(matmul PRIVATE -O3 -fmad=false -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link math library
target_link_libraries(matmul m)
