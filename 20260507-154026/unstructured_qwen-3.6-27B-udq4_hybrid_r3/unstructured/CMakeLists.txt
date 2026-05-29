cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Set CUDA architecture for RTX 3090 (compute 8.6)
set(CUDA_ARCHITECTURES 86)

# Enable CUDA language
enable_language(CUDA)

# Rename .cpp to .cu for nvcc compilation (nvcc handles both C++ and CUDA)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Build the executable
add_executable(unstructured unstructured.cpp)

# Enable optimizations for C++ and CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp")

# Link libraries: MPI, OpenMP, math
target_link_libraries(unstructured
    MPI::MPI_CXX
    -fopenmp
    -lcudart
    -lm
)

# Include directories
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_PATH})
