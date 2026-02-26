cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create the executable with CUDA support
add_executable(roomsim roomsim.cpp)

# Set CUDA architecture (adjust based on your GPU)
set_target_properties(roomsim PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Enable CUDA for the source file
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Include MPI directories
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Compiler options
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -march=native -Wall -Wextra
    >
)

# Link libraries
target_link_libraries(roomsim 
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

