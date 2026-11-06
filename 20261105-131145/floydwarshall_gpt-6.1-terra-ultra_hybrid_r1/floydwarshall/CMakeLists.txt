cmake_minimum_required(VERSION 3.18)

# Record whether to choose a default after detecting the CUDA compiler version.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(_floyd_default_cuda_architectures TRUE)
endif()

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

# Include PTX for forward compatibility and native code for common accelerators.
# The user can override this with CMAKE_CUDA_ARCHITECTURES or CUDAARCHS.
if(_floyd_default_cuda_architectures)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(_floyd_architectures 75 80 86 89 90 75-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(_floyd_architectures 60 70 75 80 86 89 90 60-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        set(_floyd_architectures 60 70 75 80 86 60-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(_floyd_architectures 60 70 75 80 60-virtual)
    else()
        set(_floyd_architectures 60 70 75 60-virtual)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_floyd_architectures}" CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# OpenMP's imported target supplies host compiler options only for CXX sources;
# this source is compiled by nvcc, so forward them to its host compiler.
separate_arguments(_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(_flag IN LISTS _openmp_flags)
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
endforeach()
