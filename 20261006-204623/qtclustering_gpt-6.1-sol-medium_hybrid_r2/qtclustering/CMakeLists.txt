cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename and executable name. This source contains
# both the CUDA kernels and MPI/OpenMP host code.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>)
# FindOpenMP's CXX compile options do not automatically apply to CUDA sources.
separate_arguments(QT_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS QT_OPENMP_FLAGS)
    target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
# Set CMAKE_CUDA_ARCHITECTURES at configure time for the cluster's GPUs.
