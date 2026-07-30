cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Force nvcc to compile the .cpp file (contains CUDA kernels)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# CUDA architecture and compilation flags
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "86"  # Ampere (RTX 30xx)
)

# Host compiler flags passed through nvcc's -Xcompiler (each flag separate)
target_compile_options(nbody PRIVATE
    -O3
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
)

# Include directories
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

# Link libraries: MPI, OpenMP, CUDA runtime, math
target_link_libraries(nbody ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} cuda m)
