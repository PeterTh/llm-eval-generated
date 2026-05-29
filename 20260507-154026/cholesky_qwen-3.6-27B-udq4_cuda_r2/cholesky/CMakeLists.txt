cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for RTX 3090 (compute 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)
target_compile_options(cholesky PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link cuBLAS, cuSOLVER, and math library
target_link_libraries(cholesky cublas cusolver m)
