cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Mark the .cpp source as CUDA so nvcc compiles it (it contains __global__ kernels)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -Xcompiler=-O3,-march=native,-Wall,-Wextra)
set_target_properties(stencil3d PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(stencil3d MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)

# Include directories
target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_PATH} ${OpenMP_CXX_INCLUDE_DIR})
