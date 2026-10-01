cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA without changing the executable/source names.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
# OpenMP's imported target supplies host libraries; nvcc needs explicit forwarding.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)
# Do not enable fast math: force summation uses IEEE double precision.
# Set CMAKE_CUDA_ARCHITECTURES at configure time to match the cluster GPUs.
