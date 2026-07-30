cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

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

# Add executable - compile as CUDA since it contains CUDA kernels
add_executable(qtclustering qtclustering.cpp)

# Set source file language to CUDA
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Include directories
target_include_directories(qtclustering PRIVATE ${CMAKE_SOURCE_DIR}/..)

# Link libraries
target_link_libraries(qtclustering 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Compile options
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-arch=sm_86>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# CUDA specific flags
set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
