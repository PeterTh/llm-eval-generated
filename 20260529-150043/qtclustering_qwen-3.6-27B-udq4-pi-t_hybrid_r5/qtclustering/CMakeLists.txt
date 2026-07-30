cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 = Ampere sm_86

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Source file - compile with nvcc for CUDA support
set(SOURCES qtclustering.cpp)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering ${SOURCES})

# Compile options for maximum performance
set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(qtclustering PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
)

# Link libraries: MPI, CUDA runtime, math
# OpenMP is linked via compile flag -fopenmp
target_link_libraries(qtclustering
    MPI::MPI_CXX
    cudart
    gomp
    m
)

# Include directories
target_include_directories(qtclustering PRIVATE
    ${MPI_CXX_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
