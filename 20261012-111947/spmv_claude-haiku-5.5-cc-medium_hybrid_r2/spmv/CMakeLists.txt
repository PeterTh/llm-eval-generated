cmake_minimum_required(VERSION 3.18)

# Default GPU targets (Volta through Hopper); override with -DCMAKE_CUDA_ARCHITECTURES=...
# Must be set before project(), which otherwise seeds the cache with a single default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# MPI, OpenMP and CUDA are all required: the benchmark always runs the hybrid version
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The CUDA kernel lives in spmv.cpp, so the translation unit is compiled by nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    # --fmad=false keeps GPU rounding identical to the host loop (no fused multiply-add)
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler;-march=native;-Xcompiler;-Wall;-Xcompiler;-Wextra;-Xcompiler;-fopenmp>)

target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)
