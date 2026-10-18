cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CXX)

# Keep double-precision arithmetic and source-order force accumulation. In
# particular, do not enable fast-math or implicit CUDA multiply-add contraction.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
