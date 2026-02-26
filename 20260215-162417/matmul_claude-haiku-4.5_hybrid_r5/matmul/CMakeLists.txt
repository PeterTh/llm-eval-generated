cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Try to find CUDA (optional)
find_package(CUDAToolkit QUIET)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link all required libraries
target_link_libraries(matmul 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m)

# Link CUDA if available
if(CUDAToolkit_FOUND)
    target_compile_definitions(matmul PRIVATE ENABLE_CUDA)
    target_link_libraries(matmul 
        CUDA::cudart
        CUDA::cublas)
endif()
