cmake_minimum_required(VERSION 3.18)

# CUDA 12.0--12.3 cannot use GCC 13 as their host compiler. Respect an
# explicitly selected compiler; otherwise use a supported GCC when available.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(CH_NVCC nvcc HINTS ENV CUDA_PATH PATH_SUFFIXES bin)
    if(CH_NVCC)
        execute_process(COMMAND "${CH_NVCC}" --version OUTPUT_VARIABLE CH_NVCC_VERSION)
        if(CH_NVCC_VERSION MATCHES "release 12\\.[0-3],")
            find_program(CH_CUDA_HOST_COMPILER NAMES g++-12 g++-11)
            if(CH_CUDA_HOST_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER "${CH_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
            endif()
        endif()
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 86 CACHE STRING "CUDA architectures")
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename while compiling its GPU kernels with nvcc.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_definitions(cahn_hilliard PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>)
# OpenMP's imported target supplies its runtime; its CXX-only compile flags
# must also be forwarded to nvcc's host compiler.
separate_arguments(CH_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(CH_OPENMP_FLAG IN LISTS CH_OPENMP_FLAGS)
    target_compile_options(cahn_hilliard PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CH_OPENMP_FLAG}>)
endforeach()
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
