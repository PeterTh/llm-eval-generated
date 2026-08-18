cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 rejects GCC 13 during compiler detection.  Prefer a compatible
# host compiler when one is available, while preserving an explicit user
# selection through CMAKE_CUDA_HOST_COMPILER.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(STENCIL3D_CUDA_HOST_COMPILER NAMES g++-12 g++-11 g++-10)
    if(STENCIL3D_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${STENCIL3D_CUDA_HOST_COMPILER}" CACHE FILEPATH
            "Host compiler used by NVCC")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # V100, A100, consumer Ampere, and H100.  The final entry also embeds PTX
    # for forward-compatible JIT compilation on newer accelerator generations.
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real 90 CACHE STRING
        "CUDA architectures for stencil3d")
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename while compiling it as CUDA C++.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

set_target_properties(stencil3d PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(stencil3d PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)
target_link_options(stencil3d PRIVATE
    -fopenmp
)
