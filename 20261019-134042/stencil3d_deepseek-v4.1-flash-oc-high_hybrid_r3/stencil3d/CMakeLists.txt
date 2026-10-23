cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Build for the common accelerator generations so the binary is portable
# across the cluster nodes.  Only architectures supported by the installed
# CUDA toolkit are enabled.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_stencil_archs 60 70 75 80 86)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
        list(APPEND _stencil_archs 89)
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
        list(APPEND _stencil_archs 90)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_stencil_archs})
endif()

add_executable(stencil3d stencil3d.cpp)

# The benchmark source contains CUDA kernels; compile it with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

target_link_libraries(stencil3d PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
)
