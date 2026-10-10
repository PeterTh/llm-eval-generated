cmake_minimum_required(VERSION 3.18)

# Avoid a legacy compiler-default GPU target and its runtime PTX compilation.
# Cluster builds can supply CMAKE_CUDA_ARCHITECTURES (or CUDAARCHS) explicitly.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24 AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Use device collectives when the MPI implementation reports CUDA support.
# Other MPI builds use pinned staging buffers with the same distributed work.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("#include <mpi.h>
#include <mpi-ext.h>
int main() { return MPIX_Query_cuda_support(); }" ROOMSIM_MPI_CUDA_QUERY)
unset(CMAKE_REQUIRED_LIBRARIES)

# Keep the original source and executable names; nvcc compiles the CUDA kernels
# in the existing .cpp file. All three parallel runtimes are required.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
if(ROOMSIM_MPI_CUDA_QUERY)
    target_compile_definitions(roomsim PRIVATE ROOMSIM_MPI_CUDA_QUERY=1)
endif()
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>
)
# OpenMP's imported target supplies the host runtime; nvcc also needs the host
# compiler's OpenMP switches explicitly.
separate_arguments(roomsim_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS roomsim_openmp_flags)
    target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
