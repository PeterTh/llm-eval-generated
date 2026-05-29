cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(MPI REQUIRED)

add_executable(nbody nbody.cu)

# Link MPI, CUDA runtime, and math
target_link_libraries(nbody
    MPI::MPI_CXX
    cuda
    cudart
    m
)

# Pass MPI include dirs to nvcc via -I flags
foreach(_inc ${MPI_CXX_INCLUDE_DIRS})
    target_compile_options(nbody PRIVATE -I${_inc})
endforeach()

# Optimisation flags for nvcc (includes OpenMP via -Xcompiler)
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "86"
)
target_compile_options(nbody PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-fopenmp
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)
target_link_options(nbody PRIVATE -fopenmp)
