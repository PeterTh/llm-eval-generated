cmake_minimum_required(VERSION 3.24)

# Override this on a build host without GPUs, or for a heterogeneous cluster,
# e.g. -DCMAKE_CUDA_ARCHITECTURES="80;90".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing filename, but compile its device kernels with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

# Do not enable fast math: the simulation uses double precision, a correctly
# rounded square root/division, and an ordered force sum for every body.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>
)
# OpenMP's imported target supplies the runtime, but its compile flags are
# restricted to CXX sources. Forward them explicitly to nvcc's host compiler.
separate_arguments(NBODY_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS NBODY_OPENMP_FLAGS)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
