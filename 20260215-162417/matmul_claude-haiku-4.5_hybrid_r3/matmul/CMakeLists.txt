cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add the matmul executable
add_executable(matmul matmul.cpp matmul_cuda.cu)

# Include directories
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Link libraries
target_link_libraries(matmul 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    m
)

# Enable optimizations
target_compile_options(matmul PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>
)

