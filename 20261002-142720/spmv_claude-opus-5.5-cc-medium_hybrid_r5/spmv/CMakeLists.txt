cmake_minimum_required(VERSION 3.24)

# Build for the GPUs present on the build machine unless specified otherwise
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# Hybrid MPI + OpenMP + CUDA: the single source is compiled with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host flags forwarded to the host compiler);
# no FP contraction on the host so the CPU reference matches the original code
target_compile_options(spmv PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,-ffp-contract=off,${OpenMP_CXX_FLAGS})

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
