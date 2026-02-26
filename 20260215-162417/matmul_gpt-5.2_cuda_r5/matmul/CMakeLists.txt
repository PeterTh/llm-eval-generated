cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Compile the existing .cpp with NVCC
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES native)

# Link CUDA runtime + cuBLAS
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cublas m)
