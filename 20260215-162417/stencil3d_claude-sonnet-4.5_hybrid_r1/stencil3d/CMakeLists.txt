cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(USE_CUDA ON)
    message(STATUS "CUDA support enabled")
else()
    set(USE_CUDA OFF)
    message(STATUS "CUDA not found - using CPU-only implementation")
endif()

add_executable(stencil3d stencil3d.cpp)

if(USE_CUDA)
    # Set CUDA properties
    set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
    
    # Enable optimizations for CUDA
    target_compile_options(stencil3d PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-fopenmp>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    )
    
    target_compile_definitions(stencil3d PRIVATE USE_CUDA)
    target_link_libraries(stencil3d CUDA::cudart)
    
    # Set CUDA architectures
    set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89 90)
else()
    # CPU-only optimizations
    target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Include directories and link libraries
target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    m
)

