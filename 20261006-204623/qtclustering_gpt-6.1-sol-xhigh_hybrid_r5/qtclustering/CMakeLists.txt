cmake_minimum_required(VERSION 3.18)

# Keep a PTX image for newer GPUs; callers may select their cluster's exact
# architectures with -DCMAKE_CUDA_ARCHITECTURES=... .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 86)
endif()
project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing file with nvcc without renaming or adding source files.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

# Disable floating-point contraction to keep CPU and GPU distance comparisons
# identical, including strict threshold boundaries and rounded-distance ties.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-ffp-contract=off,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_compile_definitions(qtclustering PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
