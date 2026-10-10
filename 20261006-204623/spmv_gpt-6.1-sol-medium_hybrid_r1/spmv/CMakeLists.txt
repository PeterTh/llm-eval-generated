cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source with nvcc without renaming it or the executable.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>)
# CMAKE_CUDA_ARCHITECTURES may be supplied for the cluster's accelerator model.
