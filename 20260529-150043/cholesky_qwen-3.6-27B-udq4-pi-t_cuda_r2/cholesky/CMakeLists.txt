cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Force the .cpp source file to be compiled with nvcc (CUDA compiler)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# CUDA architecture (RTX 3090 = sm_86)
set_target_properties(cholesky PROPERTIES
    CUDA_ARCHITECTURES 86
)

# Host compiler optimizations (passed to nvcc via -Xcompiler)
target_compile_options(cholesky PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link cuBLAS, cuSOLVER and CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky PRIVATE CUDA::cublas CUDA::cusolver CUDA::cudart)
