cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

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

# Rename the source file to .cu for CUDA compilation
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Set CUDA architectures (support common GPUs)
set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fopenmp>
)

# Include MPI
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(black_scholes 
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
)
