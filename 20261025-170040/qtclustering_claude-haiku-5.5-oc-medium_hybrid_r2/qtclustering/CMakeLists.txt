cmake_minimum_required(VERSION 3.18)

# Default GPU targets (Volta through Hopper); override with -DCMAKE_CUDA_ARCHITECTURES=...
# Must be set before project(), which otherwise picks the compiler default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The program is a single translation unit containing both host code and CUDA
# kernels, so it is compiled by nvcc.
add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# --fmad=false and -ffp-contract=off keep CPU and GPU floating-point results
# bit-identical (no fused multiply-add), so cluster decisions match exactly.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false
        -Xcompiler=-O3,-march=native,-ffp-contract=off,-fopenmp,-Wall,-Wextra>)

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)
