cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cu)

# Set CUDA architecture (adjust as needed for your GPU)
set_target_properties(spmv PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
)

# Enable optimizations
target_compile_options(spmv PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

# Link CUDA runtime
target_link_libraries(spmv PRIVATE CUDA::cudart m)
