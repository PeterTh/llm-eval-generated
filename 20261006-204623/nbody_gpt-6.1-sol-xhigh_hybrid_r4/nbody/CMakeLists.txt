cmake_minimum_required(VERSION 3.24)
# Override with -DCMAKE_CUDA_ARCHITECTURES=... for a heterogeneous cluster
# or when building on a machine without a GPU.
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

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Preserve double-precision arithmetic and ordered force accumulation. In
# particular, do not enable fast math or contraction of multiply/add pairs.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-ffp-contract=off>
)

# OpenMP's CXX imported target does not forward compile flags to nvcc's host
# compiler, so pass those flags explicitly as well as linking the target.
separate_arguments(NBODY_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS NBODY_OPENMP_FLAGS)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
