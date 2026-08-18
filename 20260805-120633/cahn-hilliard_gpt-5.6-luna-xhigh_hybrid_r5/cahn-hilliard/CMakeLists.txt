cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Native accelerator builds can override this with, for example,
    # -DCMAKE_CUDA_ARCHITECTURES=80 or 80;86;90.
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile for")
endif()

# The benchmark is intentionally built as one CUDA translation unit.  This
# keeps the existing executable name and source layout while allowing the
# MPI/OpenMP host orchestration and CUDA kernels to be compiled together.
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimized device code and optimized OpenMP host code.  The explicit
# CUDA translation unit means all three execution layers are always present.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-fopenmp;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

target_link_libraries(cahn_hilliard PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
