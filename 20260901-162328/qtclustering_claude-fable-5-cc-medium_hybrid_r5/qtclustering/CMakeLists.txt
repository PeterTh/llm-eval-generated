cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 supports host compilers up to GCC 12
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(QT_GXX12 g++-12)
    if(QT_GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${QT_GXX12})
    endif()
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# The single translation unit contains CUDA kernels; compile it with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# --fmad=false / -ffp-contract=off keep host and device distance
# computations bit-identical (required for consistent cluster selection).
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false --extended-lambda -Xcompiler=-O3,-march=native,-ffp-contract=off,-fopenmp,-Wall,-Wextra>)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link with the host toolchain (no device linking needed)
set_target_properties(qtclustering PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)
