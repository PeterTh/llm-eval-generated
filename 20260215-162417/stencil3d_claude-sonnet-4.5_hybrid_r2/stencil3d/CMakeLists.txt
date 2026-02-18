cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for CUDA availability
include(CheckLanguage)
check_language(CUDA)

add_executable(stencil3d stencil3d.cpp)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
    # Set source file to be compiled as CUDA
    set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
    
    # Set CUDA architectures
    set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
    
    # Enable optimizations
    target_compile_options(stencil3d PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --expt-relaxed-constexpr>
    )
    
    target_link_libraries(stencil3d cudart)
    target_compile_definitions(stencil3d PRIVATE USE_CUDA)
else()
    message(STATUS "CUDA not found - building CPU-only version")
    # Enable optimizations for CPU-only build
    target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries
target_link_libraries(stencil3d 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    m
)
