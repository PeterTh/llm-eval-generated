cmake_minimum_required(VERSION 3.24)

# Set CMAKE_CUDA_ARCHITECTURES explicitly when building for a different or
# heterogeneous accelerator cluster. Otherwise optimize for the local GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA without renaming it or adding files.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# The imported OpenMP target supplies host libraries; nvcc also needs the host
# compiler's OpenMP switches forwarded explicitly during compilation.
separate_arguments(stencil3d_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS stencil3d_openmp_flags)
    target_compile_options(stencil3d PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(stencil3d PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>")

# Query CUDA-aware MPI at runtime when the MPI implementation supports it.
# Other installations use asynchronous pinned-host staging of halo planes.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
    #include <mpi.h>
    #if __has_include(<mpi-ext.h>)
    #include <mpi-ext.h>
    #endif
    int main() { return MPIX_Query_cuda_support(); }
" STENCIL3D_MPI_CUDA_QUERY)
unset(CMAKE_REQUIRED_LIBRARIES)
if(STENCIL3D_MPI_CUDA_QUERY)
    target_compile_definitions(stencil3d PRIVATE STENCIL3D_MPI_CUDA_QUERY)
endif()
