cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA: pick a host compiler supported by the installed toolkit if the default
# one is too new (e.g. CUDA 12.0 does not support g++-13).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(SUPPORTED_CUDA_HOST_GXX NAMES g++-12 g++-11)
    if(SUPPORTED_CUDA_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER ${SUPPORTED_CUDA_HOST_GXX})
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# The single source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link MPI, OpenMP, CUDA runtime/cuBLAS and the math library
target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    m)
