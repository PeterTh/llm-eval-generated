cmake_minimum_required(VERSION 3.24)

# Override with -DCMAKE_CUDA_ARCHITECTURES=... for a particular cluster.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 90)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Device-buffer MPI is optional; ordinary MPI uses asynchronous pinned staging.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
    #include <mpi.h>
    #include <mpi-ext.h>
    int main() { return MPIX_Query_cuda_support(); }
" CH_MPI_CUDA_QUERY)
unset(CMAKE_REQUIRED_LIBRARIES)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra,-ffp-contract=off>)
target_compile_definitions(cahn_hilliard PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
if(CH_MPI_CUDA_QUERY)
    target_compile_definitions(cahn_hilliard PRIVATE CH_MPI_CUDA_QUERY=1)
endif()
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
