cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Rename source to .cu for CUDA compilation
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Set CUDA architecture (compute capability)
set_target_properties(nbody PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"
)

# Link libraries
target_link_libraries(nbody 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Include MPI headers
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
