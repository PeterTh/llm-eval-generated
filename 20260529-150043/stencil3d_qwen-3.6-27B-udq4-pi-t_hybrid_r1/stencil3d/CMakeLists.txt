cmake_minimum_required(VERSION 3.18)
project(stencil3d LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Enable link-time optimization for maximum performance
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

# Source file is .cpp but contains CUDA kernels — compile with nvcc
set(SOURCE_FILE stencil3d.cpp)

add_executable(stencil3d ${SOURCE_FILE})

# Set CUDA as the compiler for this target (nvcc handles both C++ and CUDA)
set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)

# Include directories
target_include_directories(stencil3d PRIVATE ../common ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Compilation flags for maximum performance
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -march=native -funroll-loops -ffast-math
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 -use_fast_math -lineinfo
        -Xcompiler -O3,-march=native,-funroll-loops,-ffast-math
        -arch=sm_86
    >
)

# Linker flags
target_link_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xlinker -O3
    >
)
