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

# Rename source file to .cu for CUDA compilation
set(SOURCE_FILE roomsim.cpp)
set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)

add_executable(roomsim ${SOURCE_FILE})

# Set CUDA architectures (adjust as needed for your GPU)
set_target_properties(roomsim PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86")

# Include directories
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Compile options
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)

# Link libraries
target_link_libraries(roomsim 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)

