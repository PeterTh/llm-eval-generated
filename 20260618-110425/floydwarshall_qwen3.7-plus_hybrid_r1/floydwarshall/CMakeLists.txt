cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Compile the .cpp file as CUDA (it contains __global__ kernels)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture for RTX 3090 (sm_86)
set_target_properties(floydwarshall PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link libraries
target_link_libraries(floydwarshall 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
