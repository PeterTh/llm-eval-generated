cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-ffp-contract=off,-Wall,-Wextra>)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
