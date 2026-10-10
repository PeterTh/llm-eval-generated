cmake_minimum_required(VERSION 3.24)

# Optimize for the build node's accelerators unless a cluster architecture list
# is supplied explicitly (for example, -DCMAKE_CUDA_ARCHITECTURES="80;90").
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# CUDA-aware MPI is optional; MPI, OpenMP and CUDA themselves are mandatory.
# Probe the MPI implementation's runtime query without assuming a vendor.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("#include <mpi.h>\nint main() { return MPIX_Query_cuda_support(); }"
    ROOMSIM_MPI_CUDA_QUERY)
if(NOT ROOMSIM_MPI_CUDA_QUERY)
    check_cxx_source_compiles("#include <mpi.h>\n#include <mpi-ext.h>\nint main() { return MPIX_Query_cuda_support(); }"
        ROOMSIM_MPI_EXT_CUDA_QUERY)
endif()
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)

# Keep the existing source filename, compiling its host and device code with nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)
if(ROOMSIM_MPI_CUDA_QUERY OR ROOMSIM_MPI_EXT_CUDA_QUERY)
    target_compile_definitions(roomsim PRIVATE ROOMSIM_MPI_CUDA_QUERY=1)
endif()
if(ROOMSIM_MPI_EXT_CUDA_QUERY)
    target_compile_definitions(roomsim PRIVATE ROOMSIM_MPI_EXT_CUDA_QUERY=1)
endif()

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-fopenmp,-ffp-contract=off,-Wall,-Wextra>)

# Required parallel runtimes and the math library.
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
