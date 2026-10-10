cmake_minimum_required(VERSION 3.24)

# Override with -DCMAKE_CUDA_ARCHITECTURES=... for heterogeneous clusters.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA without adding or renaming any files.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
target_link_libraries(floydwarshall PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
target_compile_definitions(floydwarshall PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-Wall,-Wextra>)

# OpenMP's imported target only supplies compile flags to CXX sources.
separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_host_flags)
    target_compile_options(floydwarshall PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
