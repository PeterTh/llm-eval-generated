cmake_minimum_required(VERSION 3.20)

# Cover common cluster GPUs, with PTX for newer devices. A cluster build can
# select its own architectures with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename while compiling its CUDA kernels.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Preserve double precision and the original order of each force sum.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-ffp-contract=off>
)
separate_arguments(NBODY_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS NBODY_OPENMP_FLAGS)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
