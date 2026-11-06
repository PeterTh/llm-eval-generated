cmake_minimum_required(VERSION 3.18)

# Permit a site or user to select its own GPU architectures.  Otherwise build
# for several common accelerator generations and include PTX for newer GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
   (NOT DEFINED ENV{CUDAARCHS} OR "$ENV{CUDAARCHS}" STREQUAL ""))
    set(_stencil_default_architectures TRUE)
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

if(_stencil_default_architectures)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(_stencil_architectures 75 80 90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(_stencil_architectures 60 70 80 90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(_stencil_architectures 60 70 80)
    else()
        set(_stencil_architectures 60 70)
    endif()
    # CUDA compiler detection populates this cache entry with its own default.
    # Persist our choice so a subsequent configure does not revert to that default.
    set(CMAKE_CUDA_ARCHITECTURES "${_stencil_architectures}" CACHE STRING
        "CUDA architectures for stencil3d" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The source keeps its existing name while being compiled by nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# OpenMP's imported target only applies its compile flags to C++ sources;
# pass them through nvcc to its host compiler for this CUDA translation unit.
separate_arguments(_stencil_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(_flag IN LISTS _stencil_openmp_flags)
    target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
endforeach()
