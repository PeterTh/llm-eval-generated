cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(nbody nbody.cu)

# Set CUDA architectures (adjust as needed for your cluster)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "60;70;75;80")

# Enable optimizations and warnings
target_compile_options(nbody PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
)

# Include MPI headers
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(nbody 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
