cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPUs present on the build node (matches -march=native for the host code)
set(CMAKE_CUDA_ARCHITECTURES native)

# MPI distributes row blocks across nodes, OpenMP drives GPUs within a rank, CUDA does the math
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# The source contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; host code is passed through to the host compiler
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-Wall,-Wextra,${OpenMP_CXX_FLAGS}>)

# Link math, MPI, OpenMP and CUDA runtime libraries
target_link_libraries(matmul PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
