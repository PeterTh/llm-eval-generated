cmake_minimum_required(VERSION 3.23)
project(matmul_benchmark LANGUAGES CXX)

# Honor explicit compiler and architecture selections from the environment or toolchain.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(CMAKE_CUDA_COMPILER NAMES nvcc
        HINTS "${CUDAToolkit_ROOT}" ENV CUDAToolkit_ROOT ENV CUDA_PATH /usr/local/cuda
        PATH_SUFFIXES bin)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
enable_language(CUDA)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)
# FindOpenMP's CXX flags apply to the host portion of the CUDA translation unit.
separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_host_flags)
    target_compile_options(matmul PRIVATE "-Xcompiler=${flag}")
endforeach()
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
