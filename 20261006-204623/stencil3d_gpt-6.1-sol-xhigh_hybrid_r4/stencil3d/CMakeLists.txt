cmake_minimum_required(VERSION 3.20)
# Optimize for the build host's GPUs unless a deployment architecture is supplied.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS}
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing filename, but compile its CUDA kernels with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Probe the optional MPI extension; other MPI implementations use pinned buffers
# unless CUDA-aware MPI is explicitly enabled in the runtime environment.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
    #include <mpi.h>
    #include <mpi-ext.h>
    int main() { return MPIX_Query_cuda_support(); }
" STENCIL3D_MPI_CUDA_QUERY)
unset(CMAKE_REQUIRED_LIBRARIES)
if(STENCIL3D_MPI_CUDA_QUERY)
    target_compile_definitions(stencil3d PRIVATE STENCIL3D_MPI_CUDA_QUERY=1)
endif()

target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# OpenMP's CXX target supplies the runtime at link time. Its host compilation
# flags must also be forwarded explicitly through the CUDA compiler.
separate_arguments(STENCIL3D_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS STENCIL3D_OPENMP_FLAGS)
    target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
