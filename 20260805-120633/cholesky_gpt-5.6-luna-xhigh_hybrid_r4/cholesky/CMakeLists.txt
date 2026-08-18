cmake_minimum_required(VERSION 3.24)

# Set this before project() so CMake does not initialize it to nvcc's legacy
# default (which can target an obsolete GPU).  It can still be overridden by
# -DCMAKE_CUDA_ARCHITECTURES=... for a cross-compiled or heterogeneous build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both the CUDA device code and the host code emitted
# by nvcc.  OpenMP is explicitly forwarded to the host compiler.
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)
target_compile_definitions(cholesky PRIVATE OMPI_SKIP_MPICXX)
set_target_properties(cholesky PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
