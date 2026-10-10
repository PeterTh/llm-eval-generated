cmake_minimum_required(VERSION 3.24)
# Set this before enabling CUDA, which otherwise initializes its own default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source and executable names. Users can select the cluster's
# GPU architectures with -DCMAKE_CUDA_ARCHITECTURES=... . Include PTX for newer GPUs.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
target_compile_definitions(cahn_hilliard PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# OpenMP's imported target only supplies compilation flags for CXX sources;
# forward them explicitly to nvcc's host compiler for this CUDA translation unit.
separate_arguments(CH_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS CH_OPENMP_FLAGS)
    target_compile_options(cahn_hilliard PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra,-ffp-contract=off>)
